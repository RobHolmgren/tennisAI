"""
Unit tests for the lineup deduplication logic in
tennisai/modules/lineup/predictor.py

Regression suite for the Michael Best duplicate-court bug:
the LLM was assigning the same player to both a singles and a doubles court.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLAYERS = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]


def _make_player_file(name: str):
    pf = MagicMock()
    pf.name = name
    pf.wtn_singles = 10.0
    pf.wtn_doubles = 10.0
    pf.tennisrecord_rating = 3.0
    pf.ntrp_level = 3.0
    pf.singles_wins = 3
    pf.singles_losses = 1
    pf.doubles_wins = 2
    pf.doubles_losses = 2
    pf.by_court = []
    pf.conclusions = ""
    pf.best_partners = []
    return pf


def _mock_load(name):
    return _make_player_file(name)


def _mock_assign(available, singles_courts, doubles_courts):
    """Deterministic initial lineup — no duplicates."""
    lineup = {}
    pool = list(available)
    for i in range(1, singles_courts + 1):
        lineup[f"Court {i} Singles"] = [pool.pop(0)]
    for i in range(1, doubles_courts + 1):
        if len(pool) >= 2:
            lineup[f"Court {i} Doubles"] = [pool.pop(0), pool.pop(0)]
    return lineup


def _run_predict_lineup(llm_json: dict) -> dict:
    """
    Run predict_lineup with a fully mocked LLM response.
    Returns the resulting lineup dict.
    """
    with (
        patch("tennisai.modules.lineup.predictor.load_player", side_effect=_mock_load),
        patch("tennisai.modules.lineup.predictor.assign_courts", side_effect=_mock_assign),
        patch("tennisai.modules.lineup.predictor._call_llm", return_value=json.dumps(llm_json)),
    ):
        from tennisai.modules.lineup.predictor import predict_lineup
        result = predict_lineup(
            available_players=PLAYERS,
            singles_courts=2,
            doubles_courts=3,
        )
        return result["lineup"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLineupDedup:

    def test_no_duplicate_when_llm_is_clean(self):
        """LLM assigns each player once — dedup should not change anything."""
        llm_json = {
            "lineup": [
                {"court": 1, "court_type": "Singles",  "players": ["Alice"],          "reasoning": ""},
                {"court": 2, "court_type": "Singles",  "players": ["Bob"],            "reasoning": ""},
                {"court": 1, "court_type": "Doubles",  "players": ["Carol", "Dave"],  "reasoning": ""},
                {"court": 2, "court_type": "Doubles",  "players": ["Eve", "Frank"],   "reasoning": ""},
                {"court": 3, "court_type": "Doubles",  "players": ["Grace", "Hank"],  "reasoning": ""},
            ],
            "rotation_notes": [], "season_outlook": "",
        }
        lineup = _run_predict_lineup(llm_json)
        all_players = [p for players in lineup.values() for p in players]
        assert len(all_players) == len(set(all_players)), "Unexpected duplicate"

    def test_duplicate_player_is_removed_from_second_court(self):
        """The Michael Best scenario: same player in Singles 1 and Doubles 1."""
        llm_json = {
            "lineup": [
                {"court": 1, "court_type": "Singles", "players": ["Alice"],           "reasoning": ""},
                {"court": 2, "court_type": "Singles", "players": ["Bob"],             "reasoning": ""},
                # Alice duplicated here ↓
                {"court": 1, "court_type": "Doubles", "players": ["Alice", "Carol"],  "reasoning": ""},
                {"court": 2, "court_type": "Doubles", "players": ["Dave", "Eve"],     "reasoning": ""},
                {"court": 3, "court_type": "Doubles", "players": ["Frank", "Grace"],  "reasoning": ""},
            ],
            "rotation_notes": [], "season_outlook": "",
        }
        lineup = _run_predict_lineup(llm_json)
        all_players = [p for players in lineup.values() for p in players]
        assert len(all_players) == len(set(all_players)), f"Duplicate found: {all_players}"
        # Alice should only appear once (in Singles 1)
        assert all_players.count("Alice") == 1

    def test_short_court_after_dedup_is_backfilled(self):
        """After removing Alice from Doubles 1, the court gets backfilled with an unused player."""
        llm_json = {
            "lineup": [
                {"court": 1, "court_type": "Singles", "players": ["Alice"],           "reasoning": ""},
                {"court": 2, "court_type": "Singles", "players": ["Bob"],             "reasoning": ""},
                # Alice duplicated — after dedup Doubles 1 has only Carol
                {"court": 1, "court_type": "Doubles", "players": ["Alice", "Carol"],  "reasoning": ""},
                {"court": 2, "court_type": "Doubles", "players": ["Dave", "Eve"],     "reasoning": ""},
                {"court": 3, "court_type": "Doubles", "players": ["Frank", "Grace"],  "reasoning": ""},
            ],
            "rotation_notes": [], "season_outlook": "",
        }
        lineup = _run_predict_lineup(llm_json)
        d1 = lineup.get("Court 1 Doubles", [])
        assert len(d1) == 2, f"Court 1 Doubles should have 2 players after backfill, got {d1}"

    def test_backfill_does_not_reuse_placed_players(self):
        """The player used for backfill must not already appear elsewhere."""
        llm_json = {
            "lineup": [
                {"court": 1, "court_type": "Singles", "players": ["Alice"],           "reasoning": ""},
                {"court": 2, "court_type": "Singles", "players": ["Bob"],             "reasoning": ""},
                {"court": 1, "court_type": "Doubles", "players": ["Alice", "Carol"],  "reasoning": ""},
                {"court": 2, "court_type": "Doubles", "players": ["Dave", "Eve"],     "reasoning": ""},
                {"court": 3, "court_type": "Doubles", "players": ["Frank", "Grace"],  "reasoning": ""},
            ],
            "rotation_notes": [], "season_outlook": "",
        }
        lineup = _run_predict_lineup(llm_json)
        all_players = [p for players in lineup.values() for p in players]
        assert len(all_players) == len(set(all_players))

    def test_multiple_duplicates_all_removed(self):
        """LLM puts Alice in 3 courts and Bob in 2 — all but the first occurrence are stripped."""
        llm_json = {
            "lineup": [
                {"court": 1, "court_type": "Singles", "players": ["Alice"],           "reasoning": ""},
                {"court": 2, "court_type": "Singles", "players": ["Bob"],             "reasoning": ""},
                {"court": 1, "court_type": "Doubles", "players": ["Alice", "Bob"],    "reasoning": ""},
                {"court": 2, "court_type": "Doubles", "players": ["Carol", "Alice"],  "reasoning": ""},
                {"court": 3, "court_type": "Doubles", "players": ["Dave", "Eve"],     "reasoning": ""},
            ],
            "rotation_notes": [], "season_outlook": "",
        }
        lineup = _run_predict_lineup(llm_json)
        all_players = [p for players in lineup.values() for p in players]
        assert len(all_players) == len(set(all_players))
        assert all_players.count("Alice") == 1
        assert all_players.count("Bob") == 1

    def test_player_not_in_available_list_is_rejected(self):
        """LLM hallucinates a player name not in available_players — should be excluded."""
        llm_json = {
            "lineup": [
                {"court": 1, "court_type": "Singles", "players": ["Alice"],                "reasoning": ""},
                {"court": 2, "court_type": "Singles", "players": ["PHANTOM_PLAYER"],       "reasoning": ""},
                {"court": 1, "court_type": "Doubles", "players": ["Bob", "Carol"],         "reasoning": ""},
                {"court": 2, "court_type": "Doubles", "players": ["Dave", "Eve"],          "reasoning": ""},
                {"court": 3, "court_type": "Doubles", "players": ["Frank", "Grace"],       "reasoning": ""},
            ],
            "rotation_notes": [], "season_outlook": "",
        }
        lineup = _run_predict_lineup(llm_json)
        all_players = [p for players in lineup.values() for p in players]
        assert "PHANTOM_PLAYER" not in all_players

    def test_fallback_to_initial_when_llm_returns_empty(self):
        """If LLM returns no lineup, the optimizer initial assignment is used."""
        llm_json = {"rotation_notes": [], "season_outlook": ""}  # no "lineup" key
        lineup = _run_predict_lineup(llm_json)
        # Should still have courts from the optimizer fallback
        assert len(lineup) > 0
        all_players = [p for players in lineup.values() for p in players]
        assert len(all_players) == len(set(all_players))
