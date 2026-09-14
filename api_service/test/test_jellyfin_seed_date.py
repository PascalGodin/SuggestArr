"""
Regression test for JellyfinHandler._jellyfin_item_to_seed's watch-date
extraction.

Jellyfin's /Users/{userId}/Items response reports the actual watch
timestamp under UserData.LastPlayedDate — there is no top-level
'DatePlayed' field on a real item. Checking only the top-level field name
silently fell through to DateCreated/PremiereDate for every item, which is
invisible for newly-released shows (their air date roughly tracks "recent"
anyway) but wrong for an older show watched recently — it would sort as if
watched decades ago and get dropped by the recency-based seed cap.
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from api_service.handler.jellyfin_handler import JellyfinHandler


def _make_handler():
    jellyfin_client = MagicMock()
    jellyfin_client.existing_content = {}
    tmdb_client = MagicMock()
    tmdb_client.get_metadata = AsyncMock(return_value=None)
    return JellyfinHandler(
        jellyfin_client=jellyfin_client,
        seer_client=MagicMock(),
        tmdb_client=tmdb_client,
        logger=MagicMock(),
        max_similar_movie=10,
        max_similar_tv=10,
        selected_users=[],
    )


class TestJellyfinSeedDate(unittest.IsolatedAsyncioTestCase):

    async def test_uses_user_data_last_played_date_over_premiere_date(self):
        """An old show (1999 premiere) watched moments ago must sort as
        recently watched, not as if watched in 1999."""
        item = {
            "Type": "Episode",
            "SeriesName": "One Piece",
            "SeriesProviderIds": {"Tmdb": "37854"},
            "PremiereDate": "1999-10-20T00:00:00.000Z",
            "DateCreated": "2020-01-01T00:00:00.000Z",
            "UserData": {"LastPlayedDate": "2026-07-31T22:17:00.000Z"},
        }
        handler = _make_handler()
        seed = await handler._jellyfin_item_to_seed(item, "Shows", False, {"id": "u1", "name": "Pascal"})

        expected = int(datetime(2026, 7, 31, 22, 17, tzinfo=timezone.utc).timestamp())
        self.assertIsNotNone(seed)
        self.assertEqual(seed["date"], expected)

    async def test_falls_back_to_premiere_date_when_user_data_missing(self):
        """No UserData at all (e.g. an unplayed item slipping through) still
        gets *some* date rather than crashing, via the old fallback chain."""
        item = {
            "Type": "Episode",
            "SeriesName": "Old Show",
            "SeriesProviderIds": {"Tmdb": "12345"},
            "PremiereDate": "1999-10-20T00:00:00.000Z",
        }
        handler = _make_handler()
        seed = await handler._jellyfin_item_to_seed(item, "Shows", False, {"id": "u1", "name": "Pascal"})

        self.assertIsNotNone(seed)
        self.assertGreater(seed["date"], 0)

    async def test_last_played_date_outranks_premiere_date_directly(self):
        """The actual regression: with both fields present, the resolved
        date must come from UserData, not PremiereDate."""
        recent_item = {
            "Type": "Episode",
            "SeriesName": "One Piece",
            "SeriesProviderIds": {"Tmdb": "37854"},
            "PremiereDate": "1999-10-20T00:00:00.000Z",
            "UserData": {"LastPlayedDate": "2026-07-31T22:17:00.000Z"},
        }
        premiere_only_item = {
            "Type": "Episode",
            "SeriesName": "Some New Show",
            "SeriesProviderIds": {"Tmdb": "99999"},
            "PremiereDate": "2026-07-01T00:00:00.000Z",
        }
        handler = _make_handler()
        recent_seed = await handler._jellyfin_item_to_seed(recent_item, "Shows", False, {"id": "u1", "name": "Pascal"})
        premiere_seed = await handler._jellyfin_item_to_seed(premiere_only_item, "Shows", False, {"id": "u1", "name": "Pascal"})

        self.assertGreater(
            recent_seed["date"], premiere_seed["date"],
            "A show watched tonight must outrank one merely premiering a month ago, "
            "even though its own premiere date is decades older.",
        )


if __name__ == "__main__":
    unittest.main()
