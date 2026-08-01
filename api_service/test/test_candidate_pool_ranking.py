"""
Tests for BaseMediaHandler._build_candidate_pool:

- Candidates are ranked by a TF-IDF-style genre affinity score before rating,
  so a candidate sharing the user's distinctive genres outranks a same-or-higher
  rated candidate that only shares a genre common across the whole candidate pool
  (e.g. a WWE special surfacing for a sci-fi watcher because "Action" is on
  everything).
- The job's quality filters (rating/votes incl. include_no_ratings, language,
  year, genre) are applied to the whole pool before ranking/capping.
- Excluded streaming services are checked on the final capped pool.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_service.handler.base_handler import BaseMediaHandler
from api_service.services.tmdb.tmdb_client import TMDbClient

SCIFI_GENRE = 878
ADVENTURE_GENRE = 12
ACTION_GENRE = 28


class FakeTMDbClient:
    """Minimal TMDb client stub covering only what _build_candidate_pool reads.

    Quality/streaming filters are permissive no-ops here so the ranking test
    below exercises only the genre-affinity logic in isolation.
    """

    def __init__(self, seed_map, similar_map, taste_map=None):
        self.seed_map = seed_map
        self.similar_map = similar_map
        self.taste_map = taste_map or {}
        self.tmdb_threshold = None
        self.tmdb_min_votes = None
        self.rating_source = "tmdb"
        self.language_filter = None
        self.release_year_filter = None
        self.release_year_filter_to = None
        self.genre_filter = None
        self.api_key = "fake-key"

    async def search_movie(self, title, year=None):
        item = self.seed_map.get(title)
        return [item] if item else []

    async def find_similar_movies(self, movie_id):
        return self.similar_map.get(movie_id, [])

    def _apply_filters(self, item, content_type):
        return {"passed": True}

    async def get_watch_providers(self, content_id, content_type):
        return False, None

    async def get_taste_metadata(self, content_id, content_type):
        return self.taste_map.get(content_id, {"keyword_ids": [], "keyword_names": [], "director": None})


class FakeTMDbDiscoverContext:
    def __init__(self, popular_items):
        self.popular_items = popular_items

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def discover_movies(self, filters, max_results=40):
        return self.popular_items

    async def discover_tv(self, filters, max_results=40):
        return self.popular_items


class RecordingHandler(BaseMediaHandler):
    def __init__(self, tmdb_client):
        super().__init__(
            seer_client=None,
            tmdb_client=tmdb_client,
            logger=MagicMock(),
            max_similar_movie=10,
            max_similar_tv=10,
            use_llm=True,
            dry_run=True,
        )

    def _populate_existing_content_sets(self):
        self.existing_content_sets = {}

    async def _request_llm_recommendation(self, media, item_type, source_obj, user=None):
        pass


def _item(item_id, title, genre_ids, rating, votes=100):
    return {
        "id": item_id,
        "title": title,
        "rating": rating,
        "votes": votes,
        "genre_ids": genre_ids,
    }


def _real_tmdb_client(**overrides):
    """Build a real TMDbClient (no network involved unless a test triggers it)
    so quality-filter tests exercise the production _apply_filters/get_watch_providers
    logic instead of a hand-rolled stand-in that could drift from it."""
    kwargs = dict(
        api_key="fake-key",
        search_size=40,
        # Production always coerces these to a real default (60 / 20) when
        # unset — see recommendation_automation.py — never None.
        tmdb_threshold=60,
        tmdb_min_votes=20,
        include_no_ratings=True,
        filter_release_year=0,
        filter_language=None,
        filter_genre=None,
        filter_region_provider=None,
        filter_streaming_services=None,
    )
    kwargs.update(overrides)
    client = TMDbClient(**kwargs)
    # Real network call otherwise — no keywords/director needed for these tests.
    client.get_taste_metadata = AsyncMock(
        return_value={"keyword_ids": [], "keyword_names": [], "director": None}
    )
    return client


class TestGenreAffinityRanking(unittest.IsolatedAsyncioTestCase):

    async def test_distinctive_genre_outranks_common_genre_despite_lower_rating(self):
        """A candidate sharing the seeds' rare genre beats one sharing only a
        genre so common across the pool that it carries no real signal —
        even though the common-genre candidate is rated higher."""
        seed_map = {
            "Seed1": _item(1001, "Seed1", [ACTION_GENRE, SCIFI_GENRE], 8.0),
            "Seed2": _item(1002, "Seed2", [ACTION_GENRE, ADVENTURE_GENRE], 8.0),
        }

        niche = _item(501, "Niche SciFi Movie", [SCIFI_GENRE], 7.0)
        common = _item(502, "Common Action Movie", [ACTION_GENRE], 9.0)

        similar_map = {
            1001: [niche],
            1002: [common],
        }

        # Filler "popular" items saturate the pool with the generic Action genre,
        # so its inverse-document-frequency weight collapses relative to the
        # rare Science Fiction genre carried only by `niche`.
        fillers = [
            _item(600 + i, f"Filler{i}", [ACTION_GENRE], 6.0)
            for i in range(7)
        ]

        tmdb_client = FakeTMDbClient(seed_map, similar_map)
        handler = RecordingHandler(tmdb_client)

        history_items = [
            {"title": "Seed1", "year": 2020},
            {"title": "Seed2", "year": 2021},
        ]

        with patch(
            "api_service.handler.base_handler.TMDbDiscover",
            return_value=FakeTMDbDiscoverContext(fillers),
        ):
            pool = await handler._build_candidate_pool(history_items, "movie")

        pool_ids = [c["id"] for c in pool]

        # Both come from the "recommended" (similar-items) source, so their
        # relative order is a direct read on the genre-affinity ranking itself
        # (recommended and popular candidates are ranked together, not tiered).
        self.assertLess(
            pool_ids.index(501), pool_ids.index(502),
            "Niche sci-fi candidate should outrank the common-action candidate "
            "despite its lower rating, because Action is not a distinctive "
            "genre in this candidate pool.",
        )

    async def test_popular_candidate_can_outrank_recommended_candidate(self):
        """'Recommended' (similar-to-history) and 'popular' (broad discover)
        candidates must compete on genre affinity alone — a popular candidate
        that actually matches taste should not lose out to a weaker
        'recommended' candidate just because of where it came from."""
        seed_map = {"Seed1": _item(1001, "Seed1", [SCIFI_GENRE], 8.0)}

        # No genre overlap with the seed at all — should rank low despite a
        # high rating and despite being a "recommended" candidate.
        weak_recommended = _item(501, "Weak Recommended", [], 9.0)
        similar_map = {1001: [weak_recommended]}

        # Matches the seed's genre exactly, but only surfaces as "popular".
        strong_popular = _item(502, "Strong Popular", [SCIFI_GENRE], 5.0)

        tmdb_client = FakeTMDbClient(seed_map, similar_map)
        handler = RecordingHandler(tmdb_client)

        history_items = [{"title": "Seed1", "year": 2020}]

        with patch(
            "api_service.handler.base_handler.TMDbDiscover",
            return_value=FakeTMDbDiscoverContext([strong_popular]),
        ):
            pool = await handler._build_candidate_pool(history_items, "movie")

        pool_ids = [c["id"] for c in pool]
        self.assertLess(
            pool_ids.index(502), pool_ids.index(501),
            "A popular candidate matching the seed's genre should outrank a "
            "recommended candidate with no genre overlap at all, regardless "
            "of source or rating.",
        )

    async def test_keyword_and_director_affinity_can_outrank_genre_alone(self):
        """Genre_ids are empty for every item here, so the only possible
        differentiator is a shared keyword/director — proving those signals
        genuinely feed the ranking rather than being decorative."""
        seed_map = {"Seed1": _item(1001, "Seed1", [], 8.0)}

        keyword_match = _item(501, "Keyword Match", [], 6.0)
        no_match = _item(502, "No Match", [], 9.0)
        similar_map = {1001: [keyword_match, no_match]}

        taste_map = {
            1001: {"keyword_ids": [999], "keyword_names": ["time travel"], "director": "Some Director"},
            501: {"keyword_ids": [999], "keyword_names": ["time travel"], "director": None},
            502: {"keyword_ids": [], "keyword_names": [], "director": None},
        }

        tmdb_client = FakeTMDbClient(seed_map, similar_map, taste_map=taste_map)
        handler = RecordingHandler(tmdb_client)
        history_items = [{"title": "Seed1", "year": 2020}]

        with patch(
            "api_service.handler.base_handler.TMDbDiscover",
            return_value=FakeTMDbDiscoverContext([]),
        ):
            pool = await handler._build_candidate_pool(history_items, "movie")

        pool_ids = [c["id"] for c in pool]
        self.assertLess(
            pool_ids.index(501), pool_ids.index(502),
            "Candidate sharing the seed's keyword should outrank a higher-rated "
            "candidate with no genre, keyword, or director overlap at all.",
        )

    async def test_keyword_weight_cannot_outweigh_a_full_genre_match(self):
        """Keywords are drawn from a vocabulary of thousands of values vs.
        genre's ~19, so a rare shared keyword gets a much higher raw IDF than
        a common shared genre — enough to outrank a candidate matching every
        one of the seed's genres, on IDF alone. The per-type weighting must
        keep genre matches ahead of a single incidental keyword match."""
        G1, G2, G3, G4 = 1, 2, 3, 4
        RARE_KEYWORD = 999

        seed_map = {"Seed1": _item(1001, "Seed1", [G1, G2, G3, G4], 8.0)}

        genre_match = _item(501, "Matches All Genres", [G1, G2, G3, G4], 5.0)
        keyword_match = _item(502, "Matches One Rare Keyword", [], 9.0)

        # A large filler pool sharing the same 4 genres makes them common
        # (low IDF), while the rare keyword stays unique to `keyword_match` —
        # this is the realistic-scale imbalance that surfaced in production
        # (a "texas" keyword outranking a full genre match).
        fillers = [_item(600 + i, f"Filler{i}", [G1, G2, G3, G4], 5.0) for i in range(50)]
        similar_map = {1001: [genre_match, keyword_match] + fillers}

        taste_map = {
            1001: {"keyword_ids": [RARE_KEYWORD], "keyword_names": ["rare"], "director": None},
            502: {"keyword_ids": [RARE_KEYWORD], "keyword_names": ["rare"], "director": None},
        }

        tmdb_client = FakeTMDbClient(seed_map, similar_map, taste_map=taste_map)
        handler = RecordingHandler(tmdb_client)
        history_items = [{"title": "Seed1", "year": 2020}]

        with patch(
            "api_service.handler.base_handler.TMDbDiscover",
            return_value=FakeTMDbDiscoverContext([]),
        ), patch(
            "api_service.handler.base_handler.ConfigService.get_runtime_config",
            return_value={"LLM_MAX_CANDIDATES": 100},
        ):
            pool = await handler._build_candidate_pool(history_items, "movie")

        pool_ids = [c["id"] for c in pool]
        self.assertLess(
            pool_ids.index(501), pool_ids.index(502),
            "A candidate matching all 4 of the seed's genres must outrank one "
            "matching only a single incidental keyword, even though the "
            "keyword's rarity gives it a higher raw IDF.",
        )


class TestQualityFilterIntegration(unittest.IsolatedAsyncioTestCase):

    async def test_low_rated_candidate_is_excluded_from_pool(self):
        """A candidate below the job's rating threshold never reaches the LLM,
        instead of only being caught after it's already been selected."""
        seed_map = {"Seed1": _item(1001, "Seed1", [SCIFI_GENRE], 8.0)}
        good = _item(501, "Good Match", [SCIFI_GENRE], 8.0)
        bad = _item(502, "Bad Match", [SCIFI_GENRE], 2.0)
        similar_map = {1001: [good, bad]}

        tmdb_client = _real_tmdb_client(tmdb_threshold=50, include_no_ratings=True)
        tmdb_client.search_movie = AsyncMock(side_effect=lambda title, year=None: (
            [seed_map[title]] if title in seed_map else []
        ))
        tmdb_client.find_similar_movies = AsyncMock(side_effect=lambda mid: similar_map.get(mid, []))

        handler = RecordingHandler(tmdb_client)
        history_items = [{"title": "Seed1", "year": 2020}]

        with patch(
            "api_service.handler.base_handler.TMDbDiscover",
            return_value=FakeTMDbDiscoverContext([]),
        ):
            pool = await handler._build_candidate_pool(history_items, "movie")

        pool_ids = {c["id"] for c in pool}
        self.assertIn(501, pool_ids)
        self.assertNotIn(502, pool_ids, "Candidate below the rating threshold must not reach the LLM")

    async def test_include_no_ratings_false_drops_unrated_candidates(self):
        """When the job requires a rating, an item with no vote data is dropped."""
        seed_map = {"Seed1": _item(1001, "Seed1", [SCIFI_GENRE], 8.0)}
        rated = _item(501, "Rated", [SCIFI_GENRE], 7.0)
        unrated = {"id": 502, "title": "Unrated", "genre_ids": [SCIFI_GENRE], "rating": None, "votes": None}
        similar_map = {1001: [rated, unrated]}

        tmdb_client = _real_tmdb_client(include_no_ratings=False)
        tmdb_client.search_movie = AsyncMock(side_effect=lambda title, year=None: (
            [seed_map[title]] if title in seed_map else []
        ))
        tmdb_client.find_similar_movies = AsyncMock(side_effect=lambda mid: similar_map.get(mid, []))

        handler = RecordingHandler(tmdb_client)
        history_items = [{"title": "Seed1", "year": 2020}]

        with patch(
            "api_service.handler.base_handler.TMDbDiscover",
            return_value=FakeTMDbDiscoverContext([]),
        ):
            pool = await handler._build_candidate_pool(history_items, "movie")

        pool_ids = {c["id"] for c in pool}
        self.assertIn(501, pool_ids)
        self.assertNotIn(502, pool_ids, "Unrated candidate must be dropped when include_no_ratings is False")

    async def test_streaming_excluded_candidate_is_removed_from_final_pool(self):
        """A candidate available on an excluded streaming service is dropped
        from the pool sent to the LLM."""
        seed_map = {"Seed1": _item(1001, "Seed1", [SCIFI_GENRE], 8.0)}
        keep = _item(501, "Keep Me", [SCIFI_GENRE], 8.0)
        drop = _item(502, "Drop Me", [SCIFI_GENRE], 8.0)
        similar_map = {1001: [keep, drop]}

        tmdb_client = _real_tmdb_client(
            filter_region_provider="US",
            filter_streaming_services=[{"provider_id": 8, "provider_name": "Netflix"}],
        )
        tmdb_client.search_movie = AsyncMock(side_effect=lambda title, year=None: (
            [seed_map[title]] if title in seed_map else []
        ))
        tmdb_client.find_similar_movies = AsyncMock(side_effect=lambda mid: similar_map.get(mid, []))

        async def fake_get_watch_providers(content_id, content_type):
            if content_id == 502:
                return True, "Netflix"
            return False, None

        tmdb_client.get_watch_providers = fake_get_watch_providers

        handler = RecordingHandler(tmdb_client)
        history_items = [{"title": "Seed1", "year": 2020}]

        with patch(
            "api_service.handler.base_handler.TMDbDiscover",
            return_value=FakeTMDbDiscoverContext([]),
        ):
            pool = await handler._build_candidate_pool(history_items, "movie")

        pool_ids = {c["id"] for c in pool}
        self.assertIn(501, pool_ids)
        self.assertNotIn(502, pool_ids, "Candidate on an excluded streaming service must be dropped")


if __name__ == "__main__":
    unittest.main()
