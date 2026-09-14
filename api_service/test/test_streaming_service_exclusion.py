"""
Regression tests for the streaming-service exclusion bug in
JellyfinHandler/PlexHandler.request_similar_media.

The original implementation called
``get_watch_providers(source_tmdb_obj['id'], media_type)`` — checking whether
the already-watched *seed* is on an excluded streaming service, instead of
checking each *candidate* being considered for a request. In practice this
meant the filter either excluded every candidate for a seed or none of them,
never the specific candidate that's actually on the excluded service.

These tests assert get_watch_providers is called with each candidate's own
TMDb ID, and that only the candidate flagged as excluded is skipped.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from api_service.handler.jellyfin_handler import JellyfinHandler
from api_service.handler.plex_handler import PlexHandler

SEED_ID = 999
KEEP_ID = 501
DROP_ID = 502


def _media(item_id, title):
    return {"id": item_id, "title": title, "genre_ids": []}


class FakeSeerClient:
    def __init__(self):
        self.check_already_requested = AsyncMock(return_value=False)
        self.check_already_downloaded = AsyncMock(return_value=False)
        self.check_requests_exist_batch = AsyncMock(return_value=set())
        self.request_media = AsyncMock(return_value=True)


class FakeTMDbClient:
    """Excludes only DROP_ID; records every content_id it's asked about."""

    def __init__(self):
        self.calls = []
        self.language_filter = None
        self.release_year_filter = None
        self.release_year_filter_to = None
        self.tmdb_threshold = None
        self.genre_filter = None

    async def get_watch_providers(self, content_id, content_type):
        self.calls.append(content_id)
        if content_id == DROP_ID:
            return True, "Netflix"
        return False, None

    def _apply_filters(self, item, content_type):
        return {"passed": True}


def _source():
    return {"id": SEED_ID, "title": "Seed Show"}


class TestJellyfinStreamingExclusion(unittest.IsolatedAsyncioTestCase):

    def _make_handler(self, tmdb_client, seer_client, dry_run):
        jellyfin_client = MagicMock()
        jellyfin_client.existing_content = {}
        return JellyfinHandler(
            jellyfin_client=jellyfin_client,
            seer_client=seer_client,
            tmdb_client=tmdb_client,
            logger=MagicMock(),
            max_similar_movie=10,
            max_similar_tv=10,
            selected_users=[],
            dry_run=dry_run,
        )

    async def test_dry_run_checks_candidate_id_not_seed_id(self):
        tmdb_client = FakeTMDbClient()
        handler = self._make_handler(tmdb_client, FakeSeerClient(), dry_run=True)

        await handler.request_similar_media(
            [_media(KEEP_ID, "Keep Me"), _media(DROP_ID, "Drop Me")],
            "tv", 10, _source(), user=None,
        )

        self.assertNotIn(SEED_ID, tmdb_client.calls, "Must not check the seed's own id")
        self.assertEqual(set(tmdb_client.calls), {KEEP_ID, DROP_ID})

        by_id = {item["tmdb_id"]: item for item in handler.dry_run_items}
        self.assertTrue(by_id[KEEP_ID]["would_request"])
        self.assertFalse(by_id[DROP_ID]["would_request"])

    async def test_non_dry_run_requests_only_the_non_excluded_candidate(self):
        tmdb_client = FakeTMDbClient()
        seer_client = FakeSeerClient()
        handler = self._make_handler(tmdb_client, seer_client, dry_run=False)

        await handler.request_similar_media(
            [_media(KEEP_ID, "Keep Me"), _media(DROP_ID, "Drop Me")],
            "tv", 10, _source(), user=None,
        )

        self.assertNotIn(SEED_ID, tmdb_client.calls, "Must not check the seed's own id")
        self.assertEqual(set(tmdb_client.calls), {KEEP_ID, DROP_ID})

        requested_ids = {
            call.kwargs.get("media", {}).get("id")
            for call in seer_client.request_media.await_args_list
        }
        self.assertEqual(requested_ids, {KEEP_ID})


class TestPlexStreamingExclusion(unittest.IsolatedAsyncioTestCase):

    def _make_handler(self, tmdb_client, seer_client, dry_run):
        plex_client = MagicMock()
        plex_client.existing_content = {}
        return PlexHandler(
            plex_client=plex_client,
            seer_client=seer_client,
            tmdb_client=tmdb_client,
            logger=MagicMock(),
            max_similar_movie=10,
            max_similar_tv=10,
            dry_run=dry_run,
        )

    async def test_dry_run_checks_candidate_id_not_seed_id(self):
        tmdb_client = FakeTMDbClient()
        handler = self._make_handler(tmdb_client, FakeSeerClient(), dry_run=True)

        await handler.request_similar_media(
            [_media(KEEP_ID, "Keep Me"), _media(DROP_ID, "Drop Me")],
            "movie", 10, _source(),
        )

        self.assertNotIn(SEED_ID, tmdb_client.calls, "Must not check the seed's own id")
        self.assertEqual(set(tmdb_client.calls), {KEEP_ID, DROP_ID})

        by_id = {item["tmdb_id"]: item for item in handler.dry_run_items}
        self.assertTrue(by_id[KEEP_ID]["would_request"])
        self.assertFalse(by_id[DROP_ID]["would_request"])

    async def test_non_dry_run_requests_only_the_non_excluded_candidate(self):
        tmdb_client = FakeTMDbClient()
        seer_client = FakeSeerClient()
        handler = self._make_handler(tmdb_client, seer_client, dry_run=False)

        await handler.request_similar_media(
            [_media(KEEP_ID, "Keep Me"), _media(DROP_ID, "Drop Me")],
            "movie", 10, _source(),
        )

        self.assertNotIn(SEED_ID, tmdb_client.calls, "Must not check the seed's own id")
        self.assertEqual(set(tmdb_client.calls), {KEEP_ID, DROP_ID})

        # PlexHandler's request_media call passes `media` positionally.
        requested_ids = {
            call.args[1].get("id")
            for call in seer_client.request_media.await_args_list
        }
        self.assertEqual(requested_ids, {KEEP_ID})


if __name__ == "__main__":
    unittest.main()
