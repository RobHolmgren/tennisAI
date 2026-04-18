"""
Local match history store — saves predictions and actual results to .match_history.json.
This file is gitignored and stays on the local machine only.
It feeds past prediction/result pairs back to the agent as calibration context.
"""

import datetime
import json
import uuid
from pathlib import Path
from typing import Optional

from tennisai.models import CourtResult, MatchAnalysis, MatchRecord

HISTORY_FILE = Path(__file__).parent.parent.parent / ".match_history.json"


def load_history() -> list[MatchRecord]:
    if not HISTORY_FILE.exists():
        return []
    try:
        raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return [MatchRecord(**r) for r in raw]
    except Exception:
        return []


def save_history(records: list[MatchRecord]) -> None:
    HISTORY_FILE.write_text(
        json.dumps([r.model_dump() for r in records], default=str, indent=2),
        encoding="utf-8",
    )


def record_prediction(
    analysis: MatchAnalysis,
    our_lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
) -> MatchRecord:
    """Save a new prediction to history and return the record."""
    us_wins = sum(1 for p in analysis.predictions if p.predicted_winner.lower() == "us")
    them_wins = len(analysis.predictions) - us_wins
    overall = "win" if us_wins > them_wins else ("loss" if them_wins > us_wins else "unknown")

    record = MatchRecord(
        id=str(uuid.uuid4())[:8],
        recorded_at=datetime.datetime.now().isoformat(),
        match_date=analysis.match.date,
        opponent=analysis.match.away_team or analysis.match.home_team,
        our_lineup=our_lineup,
        opponent_lineup=opponent_lineup,
        predictions=analysis.predictions,
        overall_predicted=overall,
    )

    records = load_history()
    records.append(record)
    save_history(records)
    return record


def update_result(record_id: str, actual_results: list[CourtResult]) -> Optional[MatchRecord]:
    """Fill in actual court results for a past prediction record."""
    records = load_history()
    for r in records:
        if r.id == record_id:
            r.actual_results = actual_results
            us_wins = sum(1 for c in actual_results if c.winner.lower() == "us")
            them_wins = sum(1 for c in actual_results if c.winner.lower() == "them")
            r.overall_actual = "win" if us_wins > them_wins else "loss"
            save_history(records)
            return r
    return None


def get_recent_records(n: int = 5) -> list[MatchRecord]:
    """Return the n most recent records that have actual results recorded."""
    records = load_history()
    completed = [r for r in records if r.actual_results]
    return completed[-n:]


def format_history_for_prompt(records: list[MatchRecord]) -> str:
    """Format recent match history compactly for the agent prompt."""
    if not records:
        return ""

    lines = ["Past predictions vs actuals:"]
    for r in records:
        date_str = str(r.match_date) if r.match_date else r.recorded_at[:10]
        lines.append(f"{date_str} vs {r.opponent} (predicted {r.overall_predicted}, actual {r.overall_actual or '?'})")
        for pred in r.predictions:
            actual = next(
                (a for a in r.actual_results if a.court == pred.court and a.court_type == pred.court_type),
                None,
            )
            ok = "✓" if actual and actual.winner.lower() == pred.predicted_winner.lower() else ("✗" if actual else "?")
            lines.append(
                f"  {ok} C{pred.court}{pred.court_type[0]}: pred={pred.predicted_winner}, actual={actual.winner if actual else '?'}"
            )

    return "\n".join(lines)
