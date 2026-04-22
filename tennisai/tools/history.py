"""
Backward-compatible shim — delegates to the new modules/matches and modules/players stores.
All existing callers continue to work unchanged.
"""
from typing import Optional

from tennisai.models import CourtResult, MatchAnalysis, MatchRecord, CourtPrediction
from tennisai.modules.matches.store import (
    load_all_matches,
    save_from_analysis,
    record_actual_results,
    get_recent_completed,
    migrate_from_history as _migrate,
)
from tennisai.modules.matches.models import MatchFile


# ---------------------------------------------------------------------------
# MatchRecord ↔ MatchFile bridge helpers
# ---------------------------------------------------------------------------

def _file_to_record(mf: MatchFile) -> MatchRecord:
    return MatchRecord(
        id=mf.id,
        recorded_at=mf.recorded_at,
        match_date=mf.match_date,
        opponent=mf.opponent,
        our_lineup=mf.our_lineup,
        opponent_lineup=mf.opponent_lineup,
        predictions=mf.predictions,
        actual_results=mf.actual_results,
        overall_predicted=mf.overall_predicted,
        overall_actual=mf.overall_actual,
    )


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------

def load_history() -> list[MatchRecord]:
    # Auto-migrate on first access
    _migrate()
    return [_file_to_record(mf) for mf in load_all_matches()]


def save_history(records: list[MatchRecord]) -> None:
    # No-op: individual files are saved by the matches module directly.
    pass


def record_prediction(
    analysis: MatchAnalysis,
    our_lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
) -> MatchRecord:
    mf = save_from_analysis(analysis, our_lineup, opponent_lineup)
    return _file_to_record(mf)


def update_result(record_id: str, actual_results: list[CourtResult]) -> Optional[MatchRecord]:
    mf = record_actual_results(record_id, actual_results)
    return _file_to_record(mf) if mf else None


def get_recent_records(n: int = 5) -> list[MatchRecord]:
    return [_file_to_record(mf) for mf in get_recent_completed(n)]


def format_history_for_prompt(records: list[MatchRecord]) -> str:
    if not records:
        return ""
    lines = ["Past predictions vs actuals:"]
    for r in records:
        date_str = str(r.match_date) if r.match_date else r.recorded_at[:10]
        lines.append(
            f"{date_str} vs {r.opponent} (predicted {r.overall_predicted}, actual {r.overall_actual or '?'})"
        )
        for pred in r.predictions:
            actual = next(
                (a for a in r.actual_results
                 if a.court == pred.court and a.court_type == pred.court_type),
                None,
            )
            ok = "✓" if actual and actual.winner.lower() == pred.predicted_winner.lower() else (
                "✗" if actual else "?"
            )
            lines.append(
                f"  {ok} C{pred.court}{pred.court_type[0]}: "
                f"pred={pred.predicted_winner}, actual={actual.winner if actual else '?'}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats helpers (used by agent.py run_lineup_suggestion)
# ---------------------------------------------------------------------------

def get_player_stats() -> dict[str, dict]:
    from tennisai.modules.players.store import load_all_players
    result: dict[str, dict] = {}
    for pf in load_all_players():
        result[pf.name] = {
            "matches_played": pf.matches_played,
            "singles_wins": pf.singles_wins,
            "singles_losses": pf.singles_losses,
            "doubles_wins": pf.doubles_wins,
            "doubles_losses": pf.doubles_losses,
            "by_court": {c.court_key: {"wins": c.wins, "losses": c.losses} for c in pf.by_court},
            "partners": {p.name: {"wins": p.wins, "losses": p.losses} for p in pf.best_partners},
        }
    return result


def get_season_context() -> dict:
    matches = load_all_matches()
    total = len(matches)
    completed = sum(1 for m in matches if m.actual_results)
    participation: dict[str, int] = {}
    for m in matches:
        for players in m.our_lineup.values():
            for p in players:
                participation[p] = participation.get(p, 0) + 1
    return {
        "total_recorded_matches": total,
        "completed_matches": completed,
        "player_participation": participation,
    }
