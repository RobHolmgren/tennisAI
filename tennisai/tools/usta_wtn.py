"""
WTN (World Tennis Number) lookup via the ITF public GraphQL API.

Endpoint: https://prd-itf-kube.clubspark.pro/tods-gw-api/graphql
No authentication required for publicPersons query.

WTN scale: lower = stronger (1 = elite, 40 = beginner).
Separate ratings for singles ("SINGLE") and doubles ("DOUBLE").
"""

from typing import Optional

import requests

_GQL_URL = "https://prd-itf-kube.clubspark.pro/tods-gw-api/graphql"
_QUERY = """
query SearchPlayers($term: String!) {
  publicPersons(
    pageArgs: { limit: 15, skip: 0 }
    filter: { search: { term: $term, fuzzy: true } }
  ) {
    items {
      id
      standardGivenName
      standardFamilyName
      worldTennisNumbers {
        tennisNumber
        type
        ratingDate
      }
    }
    totalItems
  }
}
"""


def _extract_wtn(world_tennis_numbers: list | None) -> dict[str, Optional[float]]:
    singles: Optional[float] = None
    doubles: Optional[float] = None
    if not world_tennis_numbers:
        return {"singles": singles, "doubles": doubles}
    for entry in world_tennis_numbers:
        t = (entry.get("type") or "").upper()
        v = entry.get("tennisNumber")
        if v is None:
            continue
        try:
            v = float(v)
        except (ValueError, TypeError):
            continue
        if not (1.0 <= v <= 40.0):
            continue
        if t == "SINGLE" and singles is None:
            singles = v
        elif t == "DOUBLE" and doubles is None:
            doubles = v
    return {"singles": singles, "doubles": doubles}


def _name_score(given: str, family: str, query: str) -> int:
    """Higher = better name match. Used to rank candidates."""
    full = f"{given} {family}".lower()
    q = query.lower()
    q_parts = q.split()
    if full == q:
        return 100
    if all(p in full for p in q_parts):
        return 80
    matched = sum(1 for p in q_parts if p in full)
    return matched * 10


def fetch_wtn_by_name(player_name: str) -> dict[str, Optional[float]]:
    """
    Look up WTN singles and doubles for a single player by name.
    Returns {"singles": float|None, "doubles": float|None}.
    """
    try:
        resp = requests.post(
            _GQL_URL,
            json={"query": _QUERY, "variables": {"term": player_name}},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [WTN] Request failed for '{player_name}': {exc}")
        return {"singles": None, "doubles": None}

    items = data.get("data", {}).get("publicPersons", {}).get("items") or []
    if not items:
        return {"singles": None, "doubles": None}

    # Pick best name match
    best = max(
        items,
        key=lambda p: _name_score(
            p.get("standardGivenName", ""),
            p.get("standardFamilyName", ""),
            player_name,
        ),
    )
    score = _name_score(
        best.get("standardGivenName", ""),
        best.get("standardFamilyName", ""),
        player_name,
    )
    if score < 10:
        return {"singles": None, "doubles": None}

    return _extract_wtn(best.get("worldTennisNumbers"))


def fetch_wtn_batch(player_names: list[str]) -> dict[str, dict[str, Optional[float]]]:
    """
    Fetch WTN for multiple players. One HTTP request per player.
    Returns {player_name: {"singles": float|None, "doubles": float|None}}.
    """
    results: dict[str, dict[str, Optional[float]]] = {}
    for name in player_names:
        results[name] = fetch_wtn_by_name(name)
    return results


def fetch_wtn(usta_profile_url: str) -> dict[str, Optional[float]]:
    """
    Legacy shim — the URL-based path is no longer needed since the ITF GraphQL
    API returns WTN by name. Returns empty result.
    """
    return {"singles": None, "doubles": None}
