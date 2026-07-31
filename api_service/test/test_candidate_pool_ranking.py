"""
Tests for the genre-affinity ranking in BaseMediaHandler._build_candidate_pool.

Candidates are ranked by a TF-IDF-style genre affinity score before rating,
so a candidate sharing the user's distinctive genres outranks a same-or-higher
rated candidate that only shares a genre common across the whole candidate pool
(e.g. a WWE special surfacing for a sci-fi watcher because "Action" is on
everything).
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_service.handler.base_handler import BaseMediaHandler

SCIFI_GENRE = 878
ADVENTURE_GENRE = 12
ACTION_GENRE = 28


class FakeTMDbClient:
    """Minimal TMDb client stub covering only what _build_candidate_pool reads."""

    def __init__(self, seed_map, similar_map):
        self.seed_map = seed_map
        self.similar_map = similar_map
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


def _item(item_id, title, genre_ids, rating):
    return {
        "id": item_id,
        "title": title,
        "rating": rating,
        "genre_ids": genre_ids,
    }


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

        # Both come from the "recommended" (similar-items) section, which is
        # ranked ahead of "popular" regardless of score — so their relative
        # order is a direct read on the genre-affinity ranking.
        self.assertLess(
            pool_ids.index(501), pool_ids.index(502),
            "Niche sci-fi candidate should outrank the common-action candidate "
            "despite its lower rating, because Action is not a distinctive "
            "genre in this candidate pool.",
        )


if __name__ == "__main__":
    unittest.main()
