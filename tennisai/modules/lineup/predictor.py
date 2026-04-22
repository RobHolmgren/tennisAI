"""
Lineup predictor: uses the optimizer for initial court assignment,
then calls the LLM to refine and add reasoning.
Works for both our team and the opponent.
"""
import json
import re

from tennisai.config import get_ai_provider, get_groq_api_key, get_anthropic_api_key, get_gemini_api_key, get_ollama_base_url, get_ollama_model
from tennisai.modules.lineup.optimizer import assign_courts
from tennisai.modules.players.store import load_player

GROQ_MODEL = "llama-3.3-70b-versatile"
CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.0-flash"


def _call_llm(prompt: str, max_tokens: int = 1024) -> str:
    provider = get_ai_provider()
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=get_groq_api_key())
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=get_anthropic_api_key())
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""
    if provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=get_gemini_api_key())
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        parts = resp.candidates[0].content.parts if resp.candidates else []
        return next((p.text for p in parts if hasattr(p, "text") and p.text), "")
    if provider == "ollama":
        from openai import OpenAI
        client = OpenAI(base_url=get_ollama_base_url(), api_key="ollama")
        resp = client.chat.completions.create(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    raise NotImplementedError(f"Provider '{provider}' not supported.")


_RATING_SCALE_NOTE = (
    "Rating scales: WTN-S = WTN singles (LOWER is stronger, 1=elite 40=beginner). "
    "WTN-D = WTN doubles (same scale). TR = tennisrecord combined (HIGHER is stronger). "
    "Use WTN-S for singles courts and WTN-D for doubles courts when available — "
    "these are more accurate than TR. Use all available signals together."
)


def _player_stat_line(name: str) -> str:
    pf = load_player(name)
    if not pf:
        return f"  {name}: no history"

    ratings: list[str] = []
    if pf.wtn_singles is not None:
        ratings.append(f"WTN-S={pf.wtn_singles}")
    if pf.wtn_doubles is not None:
        ratings.append(f"WTN-D={pf.wtn_doubles}")
    if pf.tennisrecord_rating:
        ratings.append(f"TR={pf.tennisrecord_rating}")
    elif pf.ntrp_level:
        ratings.append(f"NTRP={pf.ntrp_level}")
    rating_str = "  ".join(ratings) if ratings else "no rating"

    sr = f"{pf.singles_wins}W-{pf.singles_losses}L"
    dr = f"{pf.doubles_wins}W-{pf.doubles_losses}L"
    court_detail = ", ".join(f"{c.court_key}:{c.wins}W-{c.losses}L" for c in pf.by_court)
    record_str = f"Singles={sr} Doubles={dr}" + (f" | {court_detail}" if court_detail else "")

    if pf.conclusions:
        return f"  {name}: {rating_str} | {record_str} | {pf.conclusions}"
    return f"  {name}: {rating_str} | {record_str}"


def predict_lineup(
    available_players: list[str],
    singles_courts: int = 2,
    doubles_courts: int = 3,
    season_context: dict | None = None,
    standings: dict | None = None,
    opponent_name: str = "",
    team_label: str = "our team",
) -> dict:
    """
    Returns structured dict:
      lineup: dict[court_label, list[player_name]]
      per_court_reasoning: dict[court_label, str]
      rotation_notes: list[str]
      season_outlook: str
    Works identically for our team or the opponent (set team_label accordingly).
    """
    season = season_context or {}
    initial = assign_courts(available_players, singles_courts, doubles_courts)

    stat_lines = [_player_stat_line(p) for p in available_players]
    lineup_lines = "\n".join(
        f"  {court}: {', '.join(players)}" for court, players in initial.items()
    )

    completed = season.get("completed_matches", 0)
    total = max(season.get("total_recorded_matches", 0), completed)
    season_fraction = min(completed / max(total, 8), 1.0)

    rotation_weight = (
        "minor — focus on strongest lineup" if season_fraction < 0.4
        else "moderate — flag at-risk players" if season_fraction < 0.7
        else "important — players below 2 matches must play unless it costs the match"
    )

    rotation_flags: list[str] = []
    participation = season.get("player_participation", {})
    for player in available_players:
        played = participation.get(player, 0)
        remaining = max(total - completed, 1)
        needs = max(0, 2 - played)
        if needs > 0 and needs >= remaining:
            rotation_flags.append(f"{player}: {played} match(es), MUST play ({remaining} left)")
        elif needs > 0 and season_fraction > 0.5:
            rotation_flags.append(f"{player}: {played} match(es), at risk of missing minimum 2")

    standing_note = ""
    if standings and standings.get("our_position"):
        pos = standings["our_position"]
        ttl = standings.get("total_teams", "?")
        standing_note = f"Standings: {pos}/{ttl}. " + (
            "Playoff spot — protect it." if pos <= 2 else "Every win is critical."
        )

    court_slots_json = (
        [f'{{"court": {i}, "court_type": "Singles", "players": ["Name"], "reasoning": ""}}'
         for i in range(1, singles_courts + 1)] +
        [f'{{"court": {i}, "court_type": "Doubles", "players": ["Name1","Name2"], "reasoning": ""}}'
         for i in range(1, doubles_courts + 1)]
    )

    prompt = (
        f"Optimize the lineup for {team_label} in a USTA match vs {opponent_name or 'TBD'}.\n"
        f"{standing_note}\n"
        f"{_RATING_SCALE_NOTE}\n"
        f"IMPORTANT: Singles courts numbered 1–{singles_courts}. "
        f"Doubles courts numbered 1–{doubles_courts} (NOT 3/4/5).\n\n"
        f"Available players:\n" + "\n".join(stat_lines) + "\n\n"
        f"Initial assignment (from optimizer):\n{lineup_lines}\n\n"
        f"Rotation consideration ({int(season_fraction * 100)}% of season done): {rotation_weight}.\n"
        + (
            "Rotation flags:\n" + "\n".join(f"  {r}" for r in rotation_flags) + "\n"
            if rotation_flags else ""
        )
        + "\nAdjust if needed. Return ONLY valid JSON (no markdown):\n"
        + '{"lineup": [' + ", ".join(court_slots_json) + '], '
        + '"rotation_notes": ["..."], "season_outlook": ""}'
    )

    raw = _call_llm(prompt, max_tokens=1024)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        data = {}

    available_set = set(available_players)
    lineup: dict[str, list[str]] = {}
    per_court_reasoning: dict[str, str] = {}
    for court in data.get("lineup", []):
        label = f"Court {court['court']} {court['court_type']}"
        players = [p for p in court.get("players", []) if p in available_set]
        lineup[label] = players
        per_court_reasoning[label] = court.get("reasoning", "")

    if not lineup:
        lineup = initial

    return {
        "lineup": lineup,
        "per_court_reasoning": per_court_reasoning,
        "rotation_notes": data.get("rotation_notes", rotation_flags),
        "season_outlook": data.get("season_outlook", ""),
    }
