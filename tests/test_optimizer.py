"""
Unit tests for tennisai/modules/lineup/optimizer.py
No network calls — all player data is injected via mocks.
"""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_player(
    name: str,
    wtn_singles: float | None = None,
    wtn_doubles: float | None = None,
    tr: float | None = None,
    ntrp: float | None = None,
    singles_wins: int = 0,
    singles_losses: int = 0,
    doubles_wins: int = 0,
    doubles_losses: int = 0,
    best_partners: list | None = None,
):
    pf = MagicMock()
    pf.name = name
    pf.wtn_singles = wtn_singles
    pf.wtn_doubles = wtn_doubles
    pf.tennisrecord_rating = tr
    pf.ntrp_level = ntrp
    pf.singles_wins = singles_wins
    pf.singles_losses = singles_losses
    pf.doubles_wins = doubles_wins
    pf.doubles_losses = doubles_losses
    pf.best_partners = best_partners or []
    return pf


ROSTER_10 = [f"Player{i}" for i in range(1, 11)]

PLAYER_DATA = {
    "Player1":  _make_player("Player1",  wtn_singles=5.0,  wtn_doubles=6.0),
    "Player2":  _make_player("Player2",  wtn_singles=7.0,  wtn_doubles=8.0),
    "Player3":  _make_player("Player3",  wtn_singles=9.0,  wtn_doubles=10.0),
    "Player4":  _make_player("Player4",  wtn_singles=11.0, wtn_doubles=12.0),
    "Player5":  _make_player("Player5",  wtn_singles=13.0, wtn_doubles=14.0),
    "Player6":  _make_player("Player6",  wtn_singles=15.0, wtn_doubles=16.0),
    "Player7":  _make_player("Player7",  wtn_singles=17.0, wtn_doubles=18.0),
    "Player8":  _make_player("Player8",  wtn_singles=19.0, wtn_doubles=20.0),
    "Player9":  _make_player("Player9",  wtn_singles=21.0, wtn_doubles=22.0),
    "Player10": _make_player("Player10", wtn_singles=23.0, wtn_doubles=24.0),
}


def _mock_load_player(name):
    return PLAYER_DATA.get(name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssignCourts:

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_no_player_appears_twice(self, _mock):
        """Core regression: no player should be assigned to more than one court."""
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(ROSTER_10, singles_courts=2, doubles_courts=3)
        all_assigned = [p for players in lineup.values() for p in players]
        assert len(all_assigned) == len(set(all_assigned)), (
            f"Duplicate player found: {all_assigned}"
        )

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_court_count_matches_params(self, _mock):
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(ROSTER_10, singles_courts=2, doubles_courts=3)
        singles = [k for k in lineup if "Singles" in k]
        doubles = [k for k in lineup if "Doubles" in k]
        assert len(singles) == 2
        assert len(doubles) == 3

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_singles_courts_have_one_player(self, _mock):
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(ROSTER_10, singles_courts=2, doubles_courts=3)
        for label, players in lineup.items():
            if "Singles" in label:
                assert len(players) == 1, f"{label} should have exactly 1 player"

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_doubles_courts_have_two_players(self, _mock):
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(ROSTER_10, singles_courts=2, doubles_courts=3)
        for label, players in lineup.items():
            if "Doubles" in label:
                assert len(players) == 2, f"{label} should have exactly 2 players"

    @patch("tennisai.modules.lineup.optimizer.load_player", return_value=None)
    def test_players_without_rating_data_still_assigned(self, _mock):
        """Falls back to 0.0 score — players without files should still fill courts."""
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(ROSTER_10, singles_courts=2, doubles_courts=3)
        total_assigned = sum(len(p) for p in lineup.values())
        assert total_assigned == 8  # 2 singles + 3*2 doubles

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_stronger_players_on_lower_courts(self, _mock):
        """WTN is inverted — lower WTN = stronger. Court 1 should have the lowest WTN player."""
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(ROSTER_10, singles_courts=2, doubles_courts=3)
        s1 = lineup.get("Court 1 Singles", [])
        s2 = lineup.get("Court 2 Singles", [])
        assert s1 and s2
        wtn_s1 = PLAYER_DATA[s1[0]].wtn_singles
        wtn_s2 = PLAYER_DATA[s2[0]].wtn_singles
        assert wtn_s1 < wtn_s2, "Court 1 Singles should have a lower (stronger) WTN than Court 2"

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_too_few_players_for_doubles_handled_gracefully(self, _mock):
        """With only 3 players, we can fill 2 singles but only 0 doubles — no crash."""
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(["Player1", "Player2", "Player3"], singles_courts=2, doubles_courts=3)
        # Should not raise; doubles courts simply won't all be filled
        assert "Court 1 Singles" in lineup
        assert "Court 2 Singles" in lineup

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_only_available_players_are_assigned(self, _mock):
        """No player outside the provided list should appear in any court."""
        from tennisai.modules.lineup.optimizer import assign_courts

        available = ["Player1", "Player3", "Player5", "Player7", "Player9",
                     "Player2", "Player4", "Player6"]
        lineup = assign_courts(available, singles_courts=2, doubles_courts=3)
        available_set = set(available)
        for label, players in lineup.items():
            for p in players:
                assert p in available_set, f"{p} in {label} was not in available list"

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_single_team_format_one_singles(self, _mock):
        """Match format with 1 singles court and 3 doubles should produce correct structure."""
        from tennisai.modules.lineup.optimizer import assign_courts

        lineup = assign_courts(ROSTER_10, singles_courts=1, doubles_courts=3)
        singles = [k for k in lineup if "Singles" in k]
        doubles = [k for k in lineup if "Doubles" in k]
        assert len(singles) == 1
        assert len(doubles) == 3


class TestRankPlayers:

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_singles_ranked_by_wtn_singles(self, _mock):
        from tennisai.modules.lineup.optimizer import rank_players

        ranked = rank_players(["Player3", "Player1", "Player2"], for_singles=True)
        # Player1 has lowest WTN-S (5.0) → strongest → first
        assert ranked[0] == "Player1"
        assert ranked[-1] == "Player3"

    @patch("tennisai.modules.lineup.optimizer.load_player", side_effect=_mock_load_player)
    def test_doubles_ranked_by_wtn_doubles(self, _mock):
        from tennisai.modules.lineup.optimizer import rank_players

        ranked = rank_players(["Player3", "Player1", "Player2"], for_singles=False)
        assert ranked[0] == "Player1"
        assert ranked[-1] == "Player3"

    @patch("tennisai.modules.lineup.optimizer.load_player", return_value=None)
    def test_unknown_players_ranked_last(self, _mock):
        from tennisai.modules.lineup.optimizer import rank_players

        # All score 0.0 — order stable, no crash
        ranked = rank_players(["X", "Y", "Z"], for_singles=True)
        assert set(ranked) == {"X", "Y", "Z"}
