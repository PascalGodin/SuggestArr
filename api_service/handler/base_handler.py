"""
Base media handler with shared logic for Plex and Jellyfin handlers.
"""
import asyncio
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
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
                 seer_discovered_ids=None, dry_run=False, max_total_requests=None,
                 trakt_augmentor=None, max_content=10):
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
            trakt_augmentor: Optional MediaUserTraktAugmentor used to add Trakt
                watch-history seeds and merge fully-watched IDs into the skip set
            max_content: Max seeds to process after merging server + Trakt sources
        """
        self.seer_client = seer_client
        self.tmdb_client = tmdb_client
        self.logger = logger
        self.max_similar_movie = max_similar_movie
        self.max_similar_tv = max_similar_tv
        self.max_content = int(max_content) if max_content else 10
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
        self.trakt_augmentor = trakt_augmentor
        
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

    async def _augment_user_trakt(self, media_user_identity_id):
        """Fetch a media user's Trakt watch history additively.

        Merges fully-watched Trakt TMDB IDs into ``existing_content_sets`` (so
        they are skipped like already-owned content) and returns a list of
        normalized Trakt seed dicts (each with ``tmdb_id``, ``media_type``,
        ``title``, ``year``) for the caller to process. A missing augmentor,
        missing link, or any Trakt failure is a silent no-op returning ``[]``.
        """
        augmentor = getattr(self, "trakt_augmentor", None)
        if not augmentor or not media_user_identity_id:
            return []

        augmentation = await augmentor.augment(media_user_identity_id)
        if augmentation is None:
            return []

        total_watched = sum(len(v) for v in augmentation.watched_ids.values())
        self.logger.info(
            "Trakt: media user identity %s → %d seeds, %d watched IDs",
            media_user_identity_id, len(augmentation.seed_items), total_watched,
        )

        # Skip-watched merge: Trakt fully-watched titles join existing content.
        for media_type in ("movie", "tv"):
            watched = augmentation.watched_ids.get(media_type)
            if watched:
                self.existing_content_sets.setdefault(media_type, set()).update(watched)

        seeds = list(augmentation.seed_items)
        for seed in seeds:
            seed['source_origin'] = 'trakt_history'
            self._mark_source_origin(seed.get('source_obj'), 'trakt_history')
        return seeds

    def _merge_seeds(self, seeds):
        """Merge server and Trakt seeds, sort by date, dedup, cap to max_content.

        Each seed dict must have: ``tmdb_id``, ``media_type``, ``date`` (Unix
        timestamp). Seeds without ``date`` sort last.  Duplicate
        ``(media_type, tmdb_id)`` pairs keep the newest entry (first seen in
        descending sort).

        Returns the merged list truncated to ``self.max_content`` items.
        """
        if not seeds:
            return []

        seen = set()
        deduped = []
        for s in sorted(seeds, key=lambda x: x.get("date", 0), reverse=True):
            key = (s.get("media_type"), str(s.get("tmdb_id", "")))
            if key in seen or not s.get("tmdb_id"):
                continue
            seen.add(key)
            deduped.append(s)

        if len(deduped) > self.max_content:
            self.logger.info(
                "Merged seeds capped from %d to %d (max_content)",
                len(deduped), self.max_content,
            )
        return deduped[:self.max_content]

    def _mark_source_origin(self, source_obj, origin):
        """Attach private origin metadata to a TMDb source object."""
        if isinstance(source_obj, dict) and origin:
            source_obj['_source_origin'] = origin
        return source_obj

    @staticmethod
    def _history_key(item):
        title = str(item.get("title") or "").strip().lower()
        media_type = str(item.get("type") or item.get("media_type") or "").strip().lower()
        return (title, media_type)

    @staticmethod
    def _llm_history_item(seed):
        """Return the compact, preference-safe LLM context for one seed."""
        metadata = seed.get("source_obj") if isinstance(seed.get("source_obj"), dict) else {}
        genres = []
        for genre in metadata.get("genres") or []:
            name = genre.get("name") if isinstance(genre, dict) else genre
            if isinstance(name, str) and name.strip():
                genres.append(name.strip())

        return {
            "title": seed["title"],
            "year": seed.get("year"),
            "media_type": seed.get("media_type"),
            "genres": genres[:4],
            # A completed/recent watch is context, not proof of a preference.
            "preference_signal": seed.get("preference_signal", "recent_watch"),
            "source_origin": seed.get("source_origin"),
        }

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
    _DEFAULT_MAX_CANDIDATES = 25

    # Max simultaneous TMDb keyword/credits lookups during candidate enrichment
    # — bounded so a large eligible pool doesn't burst well past TMDb's rate limit.
    _TASTE_METADATA_CONCURRENCY = 15

    # Per-tag-type weight applied to the TF-IDF affinity score. Keywords are
    # drawn from a vocabulary of thousands of possible values vs. genre's ~19,
    # so for the same "rarity" a keyword's document frequency is structurally
    # much lower than a genre's — giving it a much higher IDF regardless of
    # whether it's actually a stronger taste signal (e.g. two shows sharing an
    # incidental "texas" setting keyword isn't as telling as sharing all of a
    # seed's genre tags). These weights damp keyword/director down so one
    # incidental shared keyword can't outrank a candidate matching every one
    # of a seed's genres; tune from observed results.
    _TAG_TYPE_WEIGHT = {'genre': 1.0, 'keyword': 0.5, 'director': 0.4}

    async def _build_candidate_pool(self, history_items: list, item_type: str) -> list:
        """Build a pool of pre-validated TMDb candidates for LLM selection.

        Resolves the top watched items to TMDb IDs, fetches similar items for
        each (reusing the same pipeline as the non-LLM path), and appends
        trending items from TMDb. The job's configured quality filters (rating
        and vote-count thresholds, include_no_ratings, language, release year,
        genre include/exclude) are applied to the whole pool before ranking, so
        candidates the job would never allow don't occupy slots that a genuinely
        matching candidate could have used. Every eligible candidate is then
        enriched with TMDb keywords and director (one combined API call each,
        bounded concurrency), and "recommended" (similar-to-history) and
        "popular" (broad discover) candidates are ranked together on equal
        footing by a combined genre/keyword/director affinity with the user's
        watch history first, rating second — this keeps the cap (env var
        LLM_MAX_CANDIDATES, default 25) from being dominated by high-rated but
        off-theme items (e.g. a niche high-rated title in a genre the user
        never watches), and from a "popular" item losing out to a weaker
        "recommended" one purely because of where it came from. Finally, the
        capped list is checked against excluded streaming services (a no-op
        network-wise when that filter isn't configured).

        Args:
            history_items: List of watched items with 'title' and 'year'.
            item_type: 'movie' or 'tv'.

        Returns:
            List of formatted TMDb result dicts ready for LLM selection.
        """
        search_fn = self.tmdb_client.search_movie if item_type == 'movie' else self.tmdb_client.search_tv
        similar_fn = self.tmdb_client.find_similar_movies if item_type == 'movie' else self.tmdb_client.find_similar_tvshows
        tc = self.tmdb_client

        # Namespaced so genre IDs, keyword IDs, and director names can never
        # collide with each other in the term/document-frequency counters
        # below (TMDb keyword IDs are arbitrary integers that could otherwise
        # coincide with a genre ID).
        def _tags_for(c) -> list:
            tags = [('genre', g) for g in (c.get('genre_ids') or [])]
            tags += [('keyword', k) for k in (c.get('keyword_ids') or [])]
            director = c.get('director')
            if director:
                tags.append(('director', director))
            return tags

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
            """Return (seed_tags, similar_items) for one watched item."""
            title = item.get('title') or item.get('name') or ''
            year = item.get('year')
            if not title:
                return [], []
            try:
                results = await search_fn(title, year)
                if not results:
                    self.logger.info("Candidate pool seed: '%s' (%s) — no TMDb search match", title, year)
                    return [], []
                matched = results[0]
                tmdb_id = matched.get('id')
                if not tmdb_id:
                    return [], []
                self.logger.info(
                    "Candidate pool seed: '%s' (%s) -> TMDb '%s' (id=%s, genre_ids=%s)",
                    title, year, matched.get('title') or matched.get('name'), tmdb_id, matched.get('genre_ids', []),
                )
                taste, similar = await asyncio.gather(
                    tc.get_taste_metadata(tmdb_id, item_type),
                    similar_fn(tmdb_id),
                )
                seed_tags = _tags_for({
                    'genre_ids': matched.get('genre_ids', []),
                    'keyword_ids': taste.get('keyword_ids', []),
                    'director': taste.get('director'),
                })
                # Attach resolved metadata to the same dict object the caller
                # passes to get_recommendations_from_history, so the "watched
                # history" prompt line gets the same rating/genre/keyword/
                # director/overview grounding the candidate list does —
                # reusing data already fetched here rather than a second round
                # trip.
                item['rating'] = matched.get('rating')
                item['genre_ids'] = matched.get('genre_ids', [])
                item['overview'] = matched.get('overview')
                item['keyword_names'] = taste.get('keyword_names', [])
                item['director'] = taste.get('director')
                return seed_tags, similar
            except Exception as exc:
                self.logger.warning("Candidate pool: error fetching similar for '%s': %s", title, exc)
                return [], []

        # Build discover filters for popular items — mirrors the job's quality settings.
        discover_filters: dict = {'sort_by': 'popularity.desc'}
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

        async def _fetch_genre_names():
            async with TMDbDiscover(tc.api_key) as tmdb_discover:
                return await tmdb_discover.get_genre_names(item_type)

        seed_results, popular, genre_name_map = await asyncio.gather(
            asyncio.gather(*[_get_similar(item) for item in history_items]),
            _fetch_popular(),
            _fetch_genre_names(),
        )

        def _attach_genre_names(item):
            genre_ids = item.get('genre_ids') or []
            item['genre_names'] = [genre_name_map[gid] for gid in genre_ids if gid in genre_name_map][:3]

        for item in history_items:
            _attach_genre_names(item)

        # Term frequency: how often each tag (genre, keyword, or director) recurs
        # across the user's watch history.
        tag_term_freq: Counter = Counter()
        similar_lists = []
        for seed_tags, similar_items in seed_results:
            tag_term_freq.update(seed_tags)
            similar_lists.append(similar_items)

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

        # Apply the job's quality filters (rating/votes incl. include_no_ratings,
        # language, release year, genre include/exclude) up front, to the whole
        # pool — not just the one item the LLM eventually picks. Without this,
        # candidates the job is configured to reject can still occupy slots in
        # the capped pool, wasting LLM attention on choices it was never allowed
        # to make.
        before_quality_filter = len(candidates)
        candidates = [c for c in candidates if tc._apply_filters(c, item_type).get('passed', True)]
        if len(candidates) != before_quality_filter:
            self.logger.debug(
                "Candidate pool: quality filters removed %d/%d candidates",
                before_quality_filter - len(candidates), before_quality_filter,
            )

        # Remove items already in the library or already discovered by Seerr.
        # These are O(1) set lookups using data loaded at handler init — no extra API calls.
        # already_requested is still checked downstream as usual; streaming-service
        # exclusion is applied later in this function, after the pool is capped.
        library_ids = self.existing_content_sets.get(item_type, set())

        def _rating(c):
            return float(c.get('rating') or c.get('vote_average') or 0)

        eligible = [
            c for c in candidates
            if _norm(c.get('title') or c.get('name') or '') not in history_titles_norm
            and str(c.get('id', '')) not in library_ids
            and not (self.honor_seer_discovery and str(c.get('id', '')) in self.seer_discovered_ids)
        ]

        # Enrich every eligible candidate with keywords and director — one
        # combined API call each, concurrency-bounded so a large pool doesn't
        # burst well past TMDb's rate limit. Quality filtering already ran
        # above, so this only pays for candidates that could actually be
        # selected.
        taste_semaphore = asyncio.Semaphore(self._TASTE_METADATA_CONCURRENCY)

        async def _enrich(c):
            async with taste_semaphore:
                taste = await tc.get_taste_metadata(c.get('id'), item_type)
            c['keyword_ids'] = taste.get('keyword_ids', [])
            c['keyword_names'] = taste.get('keyword_names', [])
            c['director'] = taste.get('director')
            _attach_genre_names(c)
            return c

        eligible = await asyncio.gather(*[_enrich(c) for c in eligible])

        # Inverse document frequency: tags shared by nearly every candidate (e.g.
        # "Drama", "Action" are on half of TMDb) are poor discriminators and get a
        # low weight; tags that only a subset of candidates carry — a shared
        # keyword or director is far rarer than a shared genre — are much more
        # telling of a genuine match and get weighted higher. Combined with term
        # frequency above, this is a lightweight TF-IDF affinity score — candidates
        # sharing the user's *distinctive* genres/keywords/director outrank
        # same-rated but off-theme results (e.g. a high-rated WWE special
        # surfacing for a sci-fi watcher).
        tag_doc_freq: Counter = Counter()
        for c in eligible:
            for t in set(_tags_for(c)):
                tag_doc_freq[t] += 1
        total_eligible = len(eligible) or 1

        def _tag_idf(t):
            return math.log((total_eligible + 1) / (tag_doc_freq.get(t, 0) + 1)) + 1

        def _tag_affinity(c):
            return sum(
                tag_term_freq.get(t, 0) * _tag_idf(t) * self._TAG_TYPE_WEIGHT.get(t[0], 1.0)
                for t in _tags_for(c)
            )

        # Rank "recommended" (similar-to-history) and "popular" (broad discover)
        # candidates on equal footing — tag affinity is the real signal we care
        # about, and a popular item that matches taste just as well as a
        # "recommended" one has no principled reason to be pushed to the back
        # just because of which TMDb endpoint it came from.
        filtered = sorted(eligible, key=lambda c: (_tag_affinity(c), _rating(c)), reverse=True)

        recommended_count = sum(1 for c in filtered if c.get('_candidate_source') == 'recommended')
        self.logger.info(
            "Candidate pool: %d items (%d recommended + %d popular, before cap)",
            len(filtered),
            recommended_count,
            len(filtered) - recommended_count,
        )
        config = ConfigService.get_runtime_config()
        max_candidates = int(config.get("LLM_MAX_CANDIDATES", self._DEFAULT_MAX_CANDIDATES))
        capped = filtered[:max_candidates]

        # Streaming-service exclusion is a per-item network call (skipped internally
        # when no region/excluded services are configured), so it's applied last,
        # only to the already-capped pool — bounded cost instead of one call per
        # raw candidate before the cap.
        async def _passes_streaming_filter(c):
            is_excluded, provider = await tc.get_watch_providers(c.get('id'), item_type)
            if is_excluded:
                self.logger.debug(
                    "Candidate pool: excluding '%s' — available on excluded service %s",
                    c.get('title') or c.get('name') or 'Unknown', provider,
                )
            return not is_excluded

        keep_flags = await asyncio.gather(*[_passes_streaming_filter(c) for c in capped])
        return [c for c, keep in zip(capped, keep_flags) if keep]

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

        trakt_history_keys = {
            self._history_key(item)
            for item in (history_items or [])
            if item.get("source_origin") == "trakt_history"
        }

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
            source_key = (str(rec.get("source_title") or "").strip().lower(), str(item_type).strip().lower())
            if source_key in trakt_history_keys:
                self._mark_source_origin(source_obj, "trakt_history")
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

            if (
                item_type == 'movie'
                and getattr(self.tmdb_client, 'only_first_movie_in_collection', False)
                and not await self.tmdb_client.is_first_movie_in_collection(best_match['id'])
            ):
                self.logger.info(
                    "Skipping LLM movie recommendation '%s': not the first movie in its collection",
                    best_match.get('title', 'Unknown'),
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
