"""
Base media handler with shared logic for Plex and Jellyfin handlers.
"""
import asyncio
import re
from abc import ABC, abstractmethod
from api_service.services.llm.llm_service import is_llm_configured, get_recommendations_from_history
from api_service.services.tmdb.tmdb_discover import TMDbDiscover
from api_service.config.config import load_env_vars
from api_service.services.config_service import ConfigService


class BaseMediaHandler(ABC):
    """
    Abstract base class for media handlers (Plex, Jellyfin).
    
    Provides shared functionality for:
    - LLM source resolution
    - LLM recommendation processing
    - Initialization of common attributes
    
    Subclasses must:
    1. Call super().__init__() with required parameters
    2. Call self._populate_existing_content_sets() after initializing their client
    3. Implement _populate_existing_content_sets() to extract client-specific content
    4. Implement _request_llm_recommendation() for handler-specific request logic
    """
    
    def __init__(self, seer_client, tmdb_client, logger,
                 max_similar_movie, max_similar_tv, library_anime_map=None,
                 use_llm=None, request_delay=0, honor_seer_discovery=False,
                 seer_discovered_ids=None, dry_run=False, max_total_requests=None):
        """
        Initialize base media handler.
        
        Args:
            seer_client: Seer service API client
            tmdb_client: TMDb API client
            logger: Logger instance
            max_similar_movie: Max number of similar movies to request
            max_similar_tv: Max number of similar TV shows to request
            library_anime_map: Dict mapping library identifiers to is_anime boolean
            use_llm: Override for LLM mode
            request_delay: Seconds to wait between consecutive requests
            honor_seer_discovery: Whether to honor Seer discovery
            seer_discovered_ids: Set of already discovered item IDs
            dry_run: Whether to simulate requests
            max_total_requests: Maximum requestable items for the whole run
        """
        self.seer_client = seer_client
        self.tmdb_client = tmdb_client
        self.logger = logger
        self.max_similar_movie = max_similar_movie
        self.max_similar_tv = max_similar_tv
        self.request_count = 0
        
        # Optimization: Pre-process existing_content into sets for O(1) lookups
        # Subclass must call _populate_existing_content_sets() after initialization
        self.existing_content_sets = {}
        
        self.library_anime_map = library_anime_map or {}
        self.request_delay = request_delay
        self.honor_seer_discovery = bool(honor_seer_discovery)
        self.seer_discovered_ids = {
            str(item_id) for item_id in (seer_discovered_ids or set())
        }
        self.dry_run = dry_run
        self.dry_run_items = []
        self._dry_run_processed_ids = set()
        self.max_total_requests = int(max_total_requests) if max_total_requests else None
        self._request_slots_reserved = 0
        self._request_limit_lock = asyncio.Lock()
        
        # Determine LLM mode
        if use_llm is not None:
            self.use_llm = use_llm
        else:
            config = load_env_vars()
            if config.get('ENABLE_ADVANCED_ALGORITHM', False):
                if is_llm_configured(config):
                    self.use_llm = True
                else:
                    self.logger.warning(
                        "ENABLE_ADVANCED_ALGORITHM is True but LLM is not configured. "
                        "AI-powered recommendations will be disabled."
                    )
                    self.use_llm = False
            else:
                self.use_llm = False

    def _has_request_capacity(self):
        """Return whether this run can still request more media."""
        return self.max_total_requests is None or self._request_slots_reserved < self.max_total_requests

    async def _reserve_request_slot(self):
        """Reserve one request slot if the job-level cap allows it."""
        if self.max_total_requests is None:
            return True

        async with self._request_limit_lock:
            if self._request_slots_reserved >= self.max_total_requests:
                return False
            self._request_slots_reserved += 1
            return True

    async def _release_request_slot(self):
        """Release a previously reserved request slot."""
        if self.max_total_requests is None:
            return

        async with self._request_limit_lock:
            self._request_slots_reserved = max(self._request_slots_reserved - 1, 0)
    
    @abstractmethod
    def _populate_existing_content_sets(self):
        """
        Populate existing_content_sets from client-specific existing content.
        
        Must be implemented by subclasses to extract existing content from 
        Plex or Jellyfin clients and convert to sets for fast lookups.
        
        Example implementation for PlexHandler:
            if self.plex_client.existing_content:
                for media_type, items in self.plex_client.existing_content.items():
                    self.existing_content_sets[media_type] = {
                        str(item.get('tmdb_id')) for item in items if item.get('tmdb_id')
                    }
        """
        pass
    
    async def _resolve_llm_source(self, source_title: str, item_type: str) -> dict:
        """
        Resolve an LLM-suggested source title to a TMDB metadata object.
        
        Strips episode notation (e.g. "Dan Da Dan - S02E12" → "Dan Da Dan") 
        before searching, so that series-level titles are looked up correctly on TMDB.
        
        Args:
            source_title: The title of the watched item that inspired the recommendation
            item_type: 'movie' or 'tv'
            
        Returns:
            TMDB metadata dict, or a fallback sentinel dict if not found
        """
        if source_title:
            # Strip episode codes like "- S02E12" or "- s02e12" that may appear in titles
            clean_title = re.sub(r'\s*[-–]\s*S\d+E\d+.*', '', source_title, flags=re.IGNORECASE).strip()
            
            if item_type == 'movie':
                results = await self.tmdb_client.search_movie(clean_title)
            else:
                results = await self.tmdb_client.search_tv(clean_title)
            
            if results:
                self.logger.debug(f"Resolved LLM source '{clean_title}' to TMDB ID {results[0].get('id')}")
                return results[0]
            
            self.logger.warning(f"Could not resolve LLM source title '{clean_title}' on TMDB.")
        
        return {"id": 0, "name": "LLM Recommendation"}
    
    # Default maximum candidates shown to the LLM — overridden by LLM_MAX_CANDIDATES env var.
    _DEFAULT_MAX_CANDIDATES = 50

    async def _build_candidate_pool(self, history_items: list, item_type: str) -> list:
        """Build a pool of pre-validated TMDb candidates for LLM selection.

        Resolves the top watched items to TMDb IDs, fetches similar items for
        each (reusing the same pipeline as the non-LLM path), and appends
        trending items from TMDb. Returns a deduplicated, filter-passing list
        capped at LLM_MAX_CANDIDATES (env var, default 50).

        Args:
            history_items: List of watched items with 'title' and 'year'.
            item_type: 'movie' or 'tv'.

        Returns:
            List of formatted TMDb result dicts ready for LLM selection.
        """
        search_fn = self.tmdb_client.search_movie if item_type == 'movie' else self.tmdb_client.search_tv
        similar_fn = self.tmdb_client.find_similar_movies if item_type == 'movie' else self.tmdb_client.find_similar_tvshows

        # Normalise history titles for membership checks (avoid recommending already-watched items).
        def _norm(title: str) -> str:
            title = re.sub(r'\s*[-–]\s*S\d+E\d+.*', '', title, flags=re.IGNORECASE)
            title = re.sub(r'\s*\((19|20)\d{2}\)\s*$', '', title)
            return title.strip().lower()

        history_titles_norm = {
            _norm(h.get('title') or h.get('name') or '')
            for h in history_items
            if h.get('title') or h.get('name')
        }

        async def _get_similar(item):
            title = item.get('title') or item.get('name') or ''
            year = item.get('year')
            if not title:
                return []
            try:
                results = await search_fn(title, year)
                if not results:
                    return []
                tmdb_id = results[0].get('id')
                if not tmdb_id:
                    return []
                return await similar_fn(tmdb_id)
            except Exception as exc:
                self.logger.warning("Candidate pool: error fetching similar for '%s': %s", title, exc)
                return []

        # Build discover filters for popular items — mirrors the job's quality settings.
        discover_filters: dict = {'sort_by': 'popularity.desc'}
        tc = self.tmdb_client
        if tc.tmdb_threshold and tc.rating_source != 'imdb':
            discover_filters['vote_average_gte'] = tc.tmdb_threshold / 10
        if tc.tmdb_min_votes and tc.rating_source != 'imdb':
            discover_filters['vote_count_gte'] = tc.tmdb_min_votes
        if tc.language_filter:
            discover_filters['with_original_language'] = tc.language_filter
        if tc.release_year_filter:
            key = 'primary_release_date_gte' if item_type == 'movie' else 'first_air_date_gte'
            discover_filters[key] = tc.release_year_filter
        if tc.release_year_filter_to:
            key = 'primary_release_date_lte' if item_type == 'movie' else 'first_air_date_lte'
            discover_filters[key] = tc.release_year_filter_to
        if tc.genre_filter:
            excluded_ids = [
                str(g.get('id')) for g in tc.genre_filter
                if isinstance(g, dict) and g.get('id')
            ]
            if excluded_ids:
                discover_filters['without_genres'] = ','.join(excluded_ids)

        async def _fetch_popular():
            async with TMDbDiscover(tc.api_key) as tmdb_discover:
                if item_type == 'movie':
                    return await tmdb_discover.discover_movies(discover_filters, max_results=40)
                return await tmdb_discover.discover_tv(discover_filters, max_results=40)

        similar_lists, popular = await asyncio.gather(
            asyncio.gather(*[_get_similar(item) for item in history_items]),
            _fetch_popular(),
        )

        # Deduplicate by TMDb ID; tag each item with its origin so the LLM prompt
        # can present them in separate labeled sections.
        seen_ids: set = set()
        candidates: list = []
        for items in similar_lists:
            for item in items:
                item_id = item.get('id')
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    item['_candidate_source'] = 'recommended'
                    candidates.append(item)
        for item in popular:
            item_id = item.get('id')
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                item['_candidate_source'] = 'popular'
                candidates.append(item)

        # Remove items already in the library or already discovered by Seerr.
        # These are O(1) set lookups using data loaded at handler init — no extra API calls.
        # Per-item checks (already_requested, watch_providers) happen downstream as usual.
        library_ids = self.existing_content_sets.get(item_type, set())

        def _rating(c):
            return float(c.get('rating') or c.get('vote_average') or 0)

        recommended_filtered = sorted(
            [c for c in candidates
             if c.get('_candidate_source') == 'recommended'
             and _norm(c.get('title') or c.get('name') or '') not in history_titles_norm
             and str(c.get('id', '')) not in library_ids
             and not (self.honor_seer_discovery and str(c.get('id', '')) in self.seer_discovered_ids)],
            key=_rating, reverse=True,
        )
        popular_filtered = sorted(
            [c for c in candidates
             if c.get('_candidate_source') == 'popular'
             and _norm(c.get('title') or c.get('name') or '') not in history_titles_norm
             and str(c.get('id', '')) not in library_ids
             and not (self.honor_seer_discovery and str(c.get('id', '')) in self.seer_discovered_ids)],
            key=_rating, reverse=True,
        )
        filtered = recommended_filtered + popular_filtered

        self.logger.info(
            "Candidate pool: %d items (%d recommended + %d popular, before cap)",
            len(filtered),
            len(recommended_filtered),
            len(popular_filtered),
        )
        config = ConfigService.get_runtime_config()
        max_candidates = int(config.get("LLM_MAX_CANDIDATES", self._DEFAULT_MAX_CANDIDATES))
        return filtered[:max_candidates]

    async def process_llm_recommendations(self, user_or_history_items, history_items_or_item_type, item_type_or_max_results, max_results=None):
        """
        Build a candidate pool from non-AI TMDb results, pass to LLM for selection,
        resolve TMDb IDs, and submit to Seer.

        Args:
            user_or_history_items: User object (new call form) or history items (legacy call form)
            history_items_or_item_type: History items (new call form) or item_type (legacy call form)
            item_type_or_max_results: Item type (new call form) or max_results (legacy call form)
            max_results: Maximum recommendations to process (new call form only)
        """
        if max_results is None:
            # Backward-compatible call form:
            # process_llm_recommendations(history_items, item_type, max_results)
            user = None
            history_items = user_or_history_items
            item_type = history_items_or_item_type
            max_results = item_type_or_max_results
        else:
            # New call form:
            # process_llm_recommendations(user, history_items, item_type, max_results)
            user = user_or_history_items
            history_items = history_items_or_item_type
            item_type = item_type_or_max_results

        if max_results <= 0:
            return

        self.logger.info(f"Delegating {max_results} {item_type} recommendations to LLM service.")

        # Build a pre-validated candidate pool so the LLM selects real items rather
        # than generating titles that may not exist on TMDb.
        candidates: list = []
        try:
            candidates = await self._build_candidate_pool(history_items, item_type)
        except Exception as exc:
            self.logger.warning(
                "Failed to build candidate pool — falling back to LLM generation mode: %s", exc
            )

        llm_recommendations = await get_recommendations_from_history(
            history_items,
            max_results,
            item_type,
            filters={
                "with_original_language": self.tmdb_client.language_filter,
                "release_year_gte": self.tmdb_client.release_year_filter,
                "release_year_lte": self.tmdb_client.release_year_filter_to,
                "vote_average_gte": self.tmdb_client.tmdb_threshold / 10 if self.tmdb_client.tmdb_threshold else None,
            },
            candidates=candidates if candidates else None,
        )

        if not llm_recommendations:
            self.logger.warning("LLM returned no recommendations.")
            return

        # Build a lookup by normalised title so matched candidates can be used
        # directly without a second TMDb search.
        candidate_lookup: dict = {}
        for c in candidates:
            title_norm = (c.get('title') or c.get('name') or '').strip().lower()
            if title_norm and title_norm not in candidate_lookup:
                candidate_lookup[title_norm] = c

        search_fn = self.tmdb_client.search_movie if item_type == 'movie' else self.tmdb_client.search_tv
        _sentinel = {"id": 0, "name": "LLM Recommendation"}

        async def resolve(rec):
            """Return (rec, [tmdb_dict], source_obj), using candidate lookup when possible."""
            title_norm = (rec.get("title") or "").strip().lower()
            matched = candidate_lookup.get(title_norm)
            if matched:
                # Candidate already validated — skip TMDb search but still resolve source.
                source_obj = await self._resolve_llm_source(rec.get("source_title"), item_type)
                return rec, [matched], source_obj

            # LLM went off-script or we are in generation mode — fall back to search.
            rec_results, source_obj = await asyncio.gather(
                search_fn(rec.get("title"), rec.get("year")),
                self._resolve_llm_source(rec.get("source_title"), item_type),
            )
            return rec, rec_results, source_obj

        resolved = await asyncio.gather(*[resolve(rec) for rec in llm_recommendations])

        request_tasks = []
        for rec, rec_results, source_obj in resolved:
            if not rec_results:
                continue

            best_match = rec_results[0]
            filter_result = self.tmdb_client._apply_filters(best_match, item_type)
            best_match['filter_results'] = filter_result

            if not filter_result.get('passed', False):
                self.logger.info(
                    "Skipping LLM %s recommendation '%s': failed configured filters (%s)",
                    item_type,
                    best_match.get('title') or best_match.get('name') or 'Unknown',
                    ', '.join(k for k, v in filter_result.items()
                              if k != 'passed' and isinstance(v, dict) and v.get('passed') is False)
                )
                continue

            best_match['rationale'] = rec.get('rationale')
            if user is None:
                request_tasks.append(self._request_llm_recommendation(best_match, item_type, source_obj))
            else:
                request_tasks.append(self._request_llm_recommendation(best_match, item_type, source_obj, user))

        if request_tasks:
            self.logger.info(f"LLM matched {len(request_tasks)} {item_type} items to TMDb.")
            await asyncio.gather(*request_tasks)
    
    @abstractmethod
    async def _request_llm_recommendation(self, media, item_type, source_obj, user=None):
        """
        Request a single LLM recommendation.
        
        Must be implemented by subclasses to handle Plex/Jellyfin-specific request logic.
        
        Args:
            media: Media item dict from TMDb search
            item_type: 'movie' or 'tv'
            source_obj: Source TMDB metadata object
            user: Optional user context for handlers that require user-specific requests
        """
        pass
