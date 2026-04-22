"""
Pure court-assignment logic — no LLM, no I/O.
Ranks players by rating + historical win rate and assigns them greedily to courts.
"""
from tennisai.modules.players.store import load_player


def rank_players(player_names: list[str], for_singles: bool) -> list[str]:
    """
    Sort players strongest-to-weakest for singles or doubles.

    Rating priority:
      1. WTN (context-specific: wtn_singles for singles, wtn_doubles for doubles)
         WTN is inverted — lower value = stronger player, so we negate it.
      2. tennisrecord_rating or ntrp_level as fallback
    Win rate blended in at 30% weight regardless of rating source.
    """
    def _score(name: str) -> float:
        pf = load_player(name)
        if not pf:
            return 0.0

        # Use the most specific available rating for this court type
        if for_singles and pf.wtn_singles is not None:
            # Negate: WTN 1 (elite) → -1, WTN 40 (beginner) → -40; sort descending = strongest first
            base = -pf.wtn_singles
            wins, losses = pf.singles_wins, pf.singles_losses
        elif not for_singles and pf.wtn_doubles is not None:
            base = -pf.wtn_doubles
            wins, losses = pf.doubles_wins, pf.doubles_losses
        else:
            base = pf.tennisrecord_rating or pf.ntrp_level or 0.0
            wins = pf.singles_wins if for_singles else pf.doubles_wins
            losses = pf.singles_losses if for_singles else pf.doubles_losses

        if (wins + losses) > 0:
            wr = wins / (wins + losses)
            return base * 0.7 + wr * 0.3
        return base

    return sorted(player_names, key=_score, reverse=True)


def best_partner_for(player: str, candidates: list[str]) -> str | None:
    """Return the historically best doubles partner from candidates."""
    pf = load_player(player)
    if not pf or not pf.best_partners:
        return candidates[0] if candidates else None
    partner_wins = {p.name: p.wins for p in pf.best_partners}
    ranked = sorted(candidates, key=lambda c: partner_wins.get(c, 0), reverse=True)
    return ranked[0] if ranked else None


def assign_courts(
    available: list[str],
    singles_courts: int,
    doubles_courts: int,
) -> dict[str, list[str]]:
    """
    Assign players to courts using ratings and match history.
    Stronger players go to lower-numbered courts.
    Doubles partners are paired by historical win rate together.
    """
    singles_ranked = rank_players(available, for_singles=True)
    doubles_ranked = rank_players(available, for_singles=False)

    lineup: dict[str, list[str]] = {}
    used: set[str] = set()

    for i in range(1, singles_courts + 1):
        for p in singles_ranked:
            if p not in used:
                lineup[f"Court {i} Singles"] = [p]
                used.add(p)
                break

    remaining = [p for p in doubles_ranked if p not in used]
    for i in range(1, doubles_courts + 1):
        if len(remaining) < 2:
            break
        p1 = remaining[0]
        partner = best_partner_for(p1, remaining[1:]) or remaining[1]
        lineup[f"Court {i} Doubles"] = [p1, partner]
        used.update({p1, partner})
        remaining = [p for p in remaining if p not in {p1, partner}]

    return lineup
