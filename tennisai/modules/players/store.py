import datetime
import re
from pathlib import Path
from typing import Optional

from tennisai.modules.players.models import CalibrationNote, CourtRecord, PartnerRecord, PlayerFile

PLAYERS_DIR = Path(__file__).parent.parent.parent.parent / "players"


def _ensure_dir() -> None:
    PLAYERS_DIR.mkdir(exist_ok=True)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def player_path(name: str) -> Path:
    return PLAYERS_DIR / f"{_slug(name)}.json"


def save_player(pf: PlayerFile) -> None:
    _ensure_dir()
    pf.updated_at = datetime.datetime.now().isoformat()
    player_path(pf.name).write_text(pf.model_dump_json(indent=2), encoding="utf-8")


def load_player(name: str) -> Optional[PlayerFile]:
    p = player_path(name)
    if not p.exists():
        return None
    try:
        return PlayerFile.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_all_players() -> list[PlayerFile]:
    _ensure_dir()
    players = []
    for p in sorted(PLAYERS_DIR.glob("*.json")):
        try:
            players.append(PlayerFile.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(players, key=lambda p: p.name)


def rebuild_from_matches() -> dict[str, PlayerFile]:
    """Rebuild all player files from completed match history. Returns updated player dict."""
    from tennisai.modules.matches.store import load_all_matches

    matches = load_all_matches()
    players: dict[str, PlayerFile] = {}

    def _ensure(name: str, team: str = "") -> PlayerFile:
        if name not in players:
            existing = load_player(name)
            if existing:
                pf = existing.model_copy()
                pf.singles_wins = 0
                pf.singles_losses = 0
                pf.doubles_wins = 0
                pf.doubles_losses = 0
                pf.matches_played = 0
                players[name] = pf
            else:
                players[name] = PlayerFile(name=name, team=team)
        return players[name]

    court_stats: dict[str, dict[str, dict]] = {}   # player → court_key → {wins, losses}
    partner_stats: dict[str, dict[str, dict]] = {}  # player → partner → {wins, losses}

    for match in matches:
        if not match.actual_results:
            continue
        for court_label, our_players in match.our_lineup.items():
            court_type = "Singles" if "Singles" in court_label else "Doubles"
            court_num = next((int(c) for c in court_label if c.isdigit()), 0)
            actual = next(
                (r for r in match.actual_results
                 if r.court == court_num and r.court_type == court_type),
                None,
            )
            if actual is None:
                continue
            won = actual.winner.lower() == "us"
            court_key = f"C{court_num}{'S' if court_type == 'Singles' else 'D'}"

            for player in our_players:
                pf = _ensure(player, match.home_team or "")
                pf.matches_played += 1
                if court_type == "Singles":
                    if won:
                        pf.singles_wins += 1
                    else:
                        pf.singles_losses += 1
                else:
                    if won:
                        pf.doubles_wins += 1
                    else:
                        pf.doubles_losses += 1

                cd = court_stats.setdefault(player, {})
                cd.setdefault(court_key, {"wins": 0, "losses": 0})
                if won:
                    cd[court_key]["wins"] += 1
                else:
                    cd[court_key]["losses"] += 1

                for partner in our_players:
                    if partner == player:
                        continue
                    pd = partner_stats.setdefault(player, {})
                    pd.setdefault(partner, {"wins": 0, "losses": 0})
                    if won:
                        pd[partner]["wins"] += 1
                    else:
                        pd[partner]["losses"] += 1

    for name, pf in players.items():
        pf.by_court = [
            CourtRecord(court_key=k, wins=v["wins"], losses=v["losses"])
            for k, v in court_stats.get(name, {}).items()
        ]
        pf.best_partners = sorted(
            [
                PartnerRecord(name=p, wins=v["wins"], losses=v["losses"])
                for p, v in partner_stats.get(name, {}).items()
            ],
            key=lambda x: x.wins,
            reverse=True,
        )
        # Preserve calibration and conclusions from existing file
        existing = load_player(name)
        if existing:
            pf.calibration = existing.calibration
            pf.conclusions = existing.conclusions
        save_player(pf)

    return players
