import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import Optional

from tennisai.models import CourtPrediction, CourtResult, MatchAnalysis
from tennisai.modules.matches.models import MatchFile

def _matches_dir() -> Path:
    from tennisai.config import get_team_config, get_team_count
    base = Path(__file__).parent.parent.parent.parent / "matches"
    if get_team_count() <= 1:
        return base
    cfg = get_team_config()
    return base / f"team-{cfg['index']}"


def _ensure_dir() -> None:
    _matches_dir().mkdir(parents=True, exist_ok=True)


def match_path(match_id: str) -> Path:
    return _matches_dir() / f"{match_id}.json"


def save_match(mf: MatchFile) -> None:
    _ensure_dir()
    match_path(mf.id).write_text(mf.model_dump_json(indent=2), encoding="utf-8")


def load_match(match_id: str) -> Optional[MatchFile]:
    p = match_path(match_id)
    if not p.exists():
        return None
    try:
        return MatchFile.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_all_matches() -> list[MatchFile]:
    _ensure_dir()
    matches = []
    for p in sorted(_matches_dir().glob("*.json")):
        try:
            matches.append(MatchFile.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(matches, key=lambda m: (m.match_date or datetime.date.min))


def save_from_analysis(
    analysis: MatchAnalysis,
    our_lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
) -> MatchFile:
    us_wins = sum(1 for p in analysis.predictions if p.predicted_winner.lower() == "us")
    them_wins = len(analysis.predictions) - us_wins
    overall = "win" if us_wins > them_wins else ("loss" if them_wins > us_wins else "unknown")

    match_id = str(uuid.uuid4())[:8]
    mf = MatchFile(
        id=match_id,
        recorded_at=datetime.datetime.now().isoformat(),
        status="upcoming",
        match_date=analysis.match.date,
        location=analysis.match.location or "",
        home_team=analysis.match.home_team or "",
        away_team=analysis.match.away_team or "",
        opponent=analysis.opponent_team.name or analysis.match.away_team or "",
        our_lineup=our_lineup,
        opponent_lineup=opponent_lineup,
        predictions=analysis.predictions,
        overall_predicted=overall,
    )
    save_match(mf)
    return mf


def record_actual_results(
    match_id: str,
    actual_results: list[CourtResult],
) -> Optional[MatchFile]:
    mf = load_match(match_id)
    if not mf:
        return None
    mf.actual_results = actual_results
    us_wins = sum(1 for c in actual_results if c.winner.lower() == "us")
    them_wins = sum(1 for c in actual_results if c.winner.lower() == "them")
    mf.overall_actual = "win" if us_wins > them_wins else "loss"
    mf.status = "completed"
    save_match(mf)
    return mf


def find_by_date_opponent(
    match_date: datetime.date,
    opponent: str,
) -> Optional[MatchFile]:
    """Return the first saved match that matches date and opponent (case-insensitive prefix)."""
    opp_key = opponent.lower()[:12]
    for mf in load_all_matches():
        if mf.match_date == match_date and opp_key in mf.opponent.lower():
            return mf
    return None


def create_stub(
    match_date: datetime.date,
    opponent: str,
    scorecard_url: str = "",
    home_team: str = "",
    away_team: str = "",
) -> MatchFile:
    """
    Create a minimal MatchFile for a past match discovered via USTA scraping.
    Uses a deterministic ID so the same match is never created twice.
    """
    det_id = hashlib.md5(f"{match_date}-{opponent.lower()}".encode()).hexdigest()[:8]
    if match_path(det_id).exists():
        existing = load_match(det_id)
        if existing:
            return existing
    mf = MatchFile(
        id=det_id,
        recorded_at=datetime.datetime.now().isoformat(),
        status="backfill-pending",
        match_date=match_date,
        opponent=opponent,
        home_team=home_team,
        away_team=away_team,
        scorecard_url=scorecard_url,
    )
    save_match(mf)
    return mf


def get_recent_completed(n: int = 5) -> list[MatchFile]:
    return [m for m in load_all_matches() if m.actual_results][-n:]


def migrate_from_history() -> int:
    """Convert .match_history.json entries into individual match files. Returns count migrated."""
    history_file = Path(__file__).parent.parent.parent.parent / ".match_history.json"
    if not history_file.exists():
        return 0
    try:
        raw = json.loads(history_file.read_text(encoding="utf-8"))
    except Exception:
        return 0

    _ensure_dir()
    count = 0
    for r in raw:
        match_id = r.get("id", str(uuid.uuid4())[:8])
        if match_path(match_id).exists():
            continue
        try:
            preds = [CourtPrediction(**p) for p in r.get("predictions", [])]
            actuals = [CourtResult(**a) for a in r.get("actual_results", [])]
            mf = MatchFile(
                id=match_id,
                recorded_at=r.get("recorded_at", datetime.datetime.now().isoformat()),
                status="completed" if actuals else "upcoming",
                match_date=r.get("match_date"),
                opponent=r.get("opponent", ""),
                our_lineup=r.get("our_lineup", {}),
                opponent_lineup=r.get("opponent_lineup", {}),
                predictions=preds,
                actual_results=actuals,
                overall_predicted=r.get("overall_predicted", ""),
                overall_actual=r.get("overall_actual"),
            )
            save_match(mf)
            count += 1
        except Exception:
            continue
    return count
