"""
Tennis AI agent — orchestrates data gathering and match prediction.

Default provider: Groq (free tier). Set AI_PROVIDER=claude in .env to use Claude instead.

Groq uses an OpenAI-compatible API: tools are defined with a "function" wrapper,
tool calls come back in message.tool_calls[], and results go back as role="tool" messages.
"""

import datetime
import json
import re
from typing import Any, Optional

from tennisai.config import get_ai_provider, get_groq_api_key, get_anthropic_api_key, get_gemini_api_key, get_ollama_base_url, get_ollama_model, get_scoring_format
from tennisai.models import CourtPrediction, Match, MatchAnalysis, Player, Team
from tennisai.tools.history import get_player_stats, get_season_context
from tennisai.tools.tennisrecord import get_league_teams, get_player_history, get_team_ratings
from tennisai.tools.usta import USTAClient

GROQ_MODEL = "llama-3.3-70b-versatile"
CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.0-flash"
MAX_ITERATIONS = 10


# ---------------------------------------------------------------------------
# Tool definitions — shared across providers
# Each entry has the raw function definition; provider wrappers add their own envelope.
# ---------------------------------------------------------------------------

_TOOL_FUNCTIONS = [
    {
        "name": "get_my_team_ratings",
        "description": (
            "Scrape player names and estimated NTRP ratings for the captain's team "
            "from a tennisrecord.com team profile URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "team_url": {"type": "string", "description": "Full tennisrecord.com team profile URL"}
            },
            "required": ["team_url"],
        },
    },
    {
        "name": "get_league_teams",
        "description": (
            "From the captain's tennisrecord.com team URL, find the league page and return "
            "all teams competing in the same flight/league (names and URLs only)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "team_url": {"type": "string", "description": "Full tennisrecord.com team profile URL"}
            },
            "required": ["team_url"],
        },
    },
    {
        "name": "get_opponent_ratings",
        "description": (
            "Scrape player names and estimated NTRP ratings for a specific opponent team "
            "from their tennisrecord.com team profile URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "team_url": {"type": "string", "description": "Full tennisrecord.com URL for the opponent team"}
            },
            "required": ["team_url"],
        },
    },
    {
        "name": "get_usta_data",
        "description": (
            "Fetch the captain's team data from USTA TennisLink: the next scheduled match "
            "(date, home team, away team, location) AND the last 5 completed match results "
            "(date, opponent, score, win/loss). Use this instead of get_next_match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "usta_team_url": {
                    "type": "string",
                    "description": "Full tennislink.usta.com Stats & Standings URL for the captain's team",
                }
            },
            "required": ["usta_team_url"],
        },
    },
    {
        "name": "get_court_trends",
        "description": (
            "Scrape individual match scorecards from USTA TennisLink to get per-court win/loss trends "
            "over the last several matches. Returns wins and losses for each court position "
            "(e.g. Singles 1, Singles 2, Doubles 1). Use this to identify which courts our team "
            "consistently wins or loses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "usta_team_url": {
                    "type": "string",
                    "description": "Full tennislink.usta.com Stats & Standings URL for the captain's team",
                }
            },
            "required": ["usta_team_url"],
        },
    },
    {
        "name": "get_player_history",
        "description": (
            "Fetch a player's individual match history from their tennisrecord.com profile page. "
            "Returns their win/loss record and recent matches over the last 6 months, including "
            "matches played outside the current team. Use this for win/loss streaks and form analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "player_url": {
                    "type": "string",
                    "description": "Full tennisrecord.com player profile URL",
                }
            },
            "required": ["player_url"],
        },
    },
]

# Groq/OpenAI format wraps each function in a {"type": "function", "function": {...}} envelope
_GROQ_TOOLS = [{"type": "function", "function": f} for f in _TOOL_FUNCTIONS]

# Anthropic format uses "input_schema" instead of "parameters", no outer envelope
_CLAUDE_TOOLS = [
    {
        "name": f["name"],
        "description": f["description"],
        "input_schema": f["parameters"],
    }
    for f in _TOOL_FUNCTIONS
]

# Gemini format: a single Tool object with a list of function declarations
_GEMINI_TOOLS = [{"function_declarations": _TOOL_FUNCTIONS}]


# ---------------------------------------------------------------------------
# Tool dispatcher (shared)
# ---------------------------------------------------------------------------

def _slim_team(team) -> dict:
    """Return team data without profile_url/team fields to save tokens."""
    return {
        "name": team.name,
        "url": team.url,
        "players": [
            {"name": p.name, "ntrp_level": p.ntrp_level, "rating": p.rating, "profile_url": p.profile_url}
            for p in team.players
        ],
    }


def _dispatch_tool(name: str, inputs: dict[str, Any]) -> str:
    if name == "get_my_team_ratings":
        return json.dumps(_slim_team(get_team_ratings(inputs["team_url"])))

    if name == "get_league_teams":
        teams = get_league_teams(inputs["team_url"])
        return json.dumps([{"name": t.name, "url": t.url} for t in teams])

    if name == "get_opponent_ratings":
        return json.dumps(_slim_team(get_team_ratings(inputs["team_url"])))

    if name == "get_usta_data":
        client = USTAClient()
        upcoming, results = client.get_schedule_and_results(inputs["usta_team_url"])
        client.close()
        next_match = next((m for m in upcoming if m.date and m.date >= datetime.date.today()), None)
        return json.dumps({
            "next_match": next_match.model_dump() if next_match else None,
            "recent_results": [r.model_dump() for r in results],
        }, default=str)

    if name == "get_court_trends":
        client = USTAClient()
        trends = client.get_court_trends(inputs["usta_team_url"])
        client.close()
        return json.dumps([t.model_dump() for t in trends])

    if name == "get_player_history":
        history = get_player_history(inputs["player_url"])
        return history.model_dump_json()

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    scoring = get_scoring_format()
    return (
        "You are a USTA adult league tennis analyst. Use tools to gather data, then predict court outcomes.\n\n"
        f"Scoring format: {scoring}. "
        "Predict scores like '6-3 6-2' or '4-6 6-3 [10-7]'. First number in each set = our score.\n\n"
        "Rating scales when provided:\n"
        "  WTN-S = WTN singles (LOWER is stronger: 1=elite, 40=beginner). Use for singles courts.\n"
        "  WTN-D = WTN doubles (same scale). Use for doubles courts.\n"
        "  TR = tennisrecord combined rating (HIGHER is stronger, e.g. 3.12 beats 2.94).\n"
        "Use WTN-S/WTN-D as primary signals when available; TR as supporting context. "
        "Use all available signals together — never ignore a rating just because another is present.\n\n"
        "Weigh these factors: (1) ratings, (2) per-court win/loss trends, (3) player form last 6 months, "
        "(4) predicted opponent lineup strongest-to-weakest by court, (5) lineup changes to maximise team wins. "
        "Goal is team match victory (most courts), not individual courts. "
        "Court 1 > Court 2 > Court 3 in difficulty and importance — put stronger players lower-numbered.\n\n"
        "Provide final analysis without asking follow-up questions."
    )


_SYSTEM_PROMPT = _build_system_prompt()

_OUTPUT_SCHEMA = (
    "Return ONLY a JSON object. JSON rules: double-quotes for all strings and keys, "
    "square brackets [] for every array (never parentheses or curly braces), "
    "no trailing commas, no comments.\n\n"
    "{\n"
    '  "match": {"date": null, "home_team": "", "away_team": "", "location": ""},\n'
    '  "my_team": {"name": "", "url": ""},\n'
    '  "opponent_team": {"name": "", "url": ""},\n'
    '  "predictions": [\n'
    '    {\n'
    '      "court": 1,\n'
    '      "court_type": "Singles",\n'
    '      "my_players": ["Player A"],\n'
    '      "opponent_players": ["Player B"],\n'
    '      "predicted_winner": "us",\n'
    '      "predicted_score": "6-3 6-2",\n'
    '      "confidence": "high",\n'
    '      "reasoning": "one sentence"\n'
    '    }\n'
    "  ],\n"
    '  "overall_outlook": "one paragraph",\n'
    '  "lineup_suggestions": ["suggestion 1", "suggestion 2"]\n'
    "}"
)


def _build_user_message(
    my_team_url: str,
    usta_team_url: str,
    lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
    history_text: str,
    singles_courts: int = 2,
    doubles_courts: int = 3,
) -> str:
    lineup_text = "\n".join(f"  {court}: {', '.join(players)}" for court, players in lineup.items())

    opponent_section = ""
    if opponent_lineup:
        opp_text = "\n".join(f"  {court}: {', '.join(players)}" for court, players in opponent_lineup.items())
        opponent_section = (
            f"The opponent's confirmed lineup (use this instead of predicting their lineup):\n{opp_text}\n\n"
        )

    history_section = f"\n{history_text}\n\n" if history_text else ""

    opponent_lineup_step = (
        "7. The opponent's lineup is already known (provided above) — use it directly, skip prediction step\n"
        if opponent_lineup else
        "7. Predict the opponent's most likely lineup by ranking their players strongest-to-weakest "
        "and assigning them to courts accordingly\n"
    )

    singles_labels = " | ".join(f"Singles court={i}" for i in range(1, singles_courts + 1))
    doubles_labels = " | ".join(f"Doubles court={i}" for i in range(1, doubles_courts + 1))
    court_format = (
        f"MATCH FORMAT: {singles_courts} Singles courts + {doubles_courts} Doubles courts = "
        f"{singles_courts + doubles_courts} courts total. Courts: [{singles_labels} | {doubles_labels}]. "
        "Do NOT invent other courts. Doubles courts are numbered 1, 2, 3 — NOT 3, 4, 5."
    )
    winner_note = (
        "CRITICAL — predicted_winner field: write 'us' when THE CAPTAIN'S TEAM wins that court, "
        "write 'them' when THE OPPONENT wins. Never write 'them' for a court our team is predicted to win."
    )

    return (
        f"Analyze our next match. tennisrecord URL: {my_team_url} | USTA URL: {usta_team_url}\n\n"
        f"{court_format}\n\n"
        f"{winner_note}\n\n"
        f"Our lineup:\n{lineup_text}\n\n"
        f"{opponent_section}"
        f"{history_section}"
        "Steps: 1) get_my_team_ratings 2) get_usta_data 3) get_court_trends "
        "4) get_league_teams then get_opponent_ratings 5) get_player_history for each of our players "
        f"6) {opponent_lineup_step.strip()} "
        "7) predict each court with score 8) suggest lineup changes 9) overall outlook\n\n"
        + _OUTPUT_SCHEMA
    )


# ---------------------------------------------------------------------------
# Groq agent loop
# ---------------------------------------------------------------------------

def _run_groq(
    my_team_url: str,
    usta_team_url: str,
    lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
    history_text: str,
    singles_courts: int,
    doubles_courts: int,
) -> MatchAnalysis:
    from groq import Groq

    client = Groq(api_key=get_groq_api_key())
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(my_team_url, usta_team_url, lineup, opponent_lineup, history_text, singles_courts, doubles_courts)},
    ]

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=_GROQ_TOOLS,
            tool_choice="auto",
            max_tokens=4096,
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return _parse_final_response(message.content or "", my_team_url, usta_team_url)

        for tool_call in message.tool_calls:
            try:
                result = _dispatch_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                )
            except Exception as exc:
                result = json.dumps({"error": str(exc)})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    raise RuntimeError("Agent exceeded maximum iterations without producing a final answer.")


# ---------------------------------------------------------------------------
# Claude agent loop
# ---------------------------------------------------------------------------

def _run_claude(
    my_team_url: str,
    usta_team_url: str,
    lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
    history_text: str,
    singles_courts: int,
    doubles_courts: int,
) -> MatchAnalysis:
    import anthropic

    client = anthropic.Anthropic(api_key=get_anthropic_api_key())
    messages: list[dict] = [
        {"role": "user", "content": _build_user_message(my_team_url, usta_team_url, lineup, opponent_lineup, history_text, singles_courts, doubles_courts)},
    ]

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=_CLAUDE_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")), ""
            )
            return _parse_final_response(final_text, my_team_url, usta_team_url)

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = _dispatch_tool(block.name, block.input)
                    except Exception as exc:
                        result = json.dumps({"error": str(exc)})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

    raise RuntimeError("Agent exceeded maximum iterations without producing a final answer.")


# ---------------------------------------------------------------------------
# Ollama agent loop (OpenAI-compatible, no API key needed)
# ---------------------------------------------------------------------------

def _run_ollama(
    my_team_url: str,
    usta_team_url: str,
    lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
    history_text: str,
    singles_courts: int,
    doubles_courts: int,
) -> MatchAnalysis:
    from openai import OpenAI

    client = OpenAI(base_url=get_ollama_base_url(), api_key="ollama")
    model = get_ollama_model()
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(
            my_team_url, usta_team_url, lineup, opponent_lineup,
            history_text, singles_courts, doubles_courts,
        )},
    ]

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_GROQ_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return _parse_final_response(message.content or "", my_team_url, usta_team_url)

        for tool_call in message.tool_calls:
            try:
                result = _dispatch_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                )
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    raise RuntimeError("Agent exceeded maximum iterations without producing a final answer.")


# ---------------------------------------------------------------------------
# Gemini agent loop
# ---------------------------------------------------------------------------

def _run_gemini(
    my_team_url: str,
    usta_team_url: str,
    lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
    history_text: str,
    singles_courts: int,
    doubles_courts: int,
) -> MatchAnalysis:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_gemini_api_key())
    user_msg = _build_user_message(
        my_team_url, usta_team_url, lineup, opponent_lineup,
        history_text, singles_courts, doubles_courts,
    )
    contents: list[dict] = [{"role": "user", "parts": [{"text": user_msg}]}]
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        tools=_GEMINI_TOOLS,
        max_output_tokens=4096,
    )

    for _ in range(MAX_ITERATIONS):
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        parts = response.candidates[0].content.parts
        model_parts: list[dict] = []
        tool_calls: list = []

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                model_parts.append({"function_call": {"name": fc.name, "args": dict(fc.args)}})
                tool_calls.append(fc)
            elif hasattr(part, "text") and part.text:
                model_parts.append({"text": part.text})

        contents.append({"role": "model", "parts": model_parts})

        if not tool_calls:
            text = next((p["text"] for p in model_parts if "text" in p), "")
            return _parse_final_response(text, my_team_url, usta_team_url)

        result_parts: list[dict] = []
        for fc in tool_calls:
            try:
                result = _dispatch_tool(fc.name, dict(fc.args))
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
            result_parts.append({
                "function_response": {"name": fc.name, "response": {"result": result}}
            })
        contents.append({"role": "user", "parts": result_parts})

    raise RuntimeError("Agent exceeded maximum iterations without producing a final answer.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analysis(
    my_team_url: str,
    usta_team_url: str,
    lineup: dict[str, list[str]],
    opponent_lineup: Optional[dict[str, list[str]]] = None,
    history_text: str = "",
    singles_courts: int = 2,
    doubles_courts: int = 3,
) -> MatchAnalysis:
    opp = opponent_lineup or {}
    provider = get_ai_provider()
    if provider == "groq":
        return _run_groq(my_team_url, usta_team_url, lineup, opp, history_text, singles_courts, doubles_courts)
    if provider == "claude":
        return _run_claude(my_team_url, usta_team_url, lineup, opp, history_text, singles_courts, doubles_courts)
    if provider == "gemini":
        return _run_gemini(my_team_url, usta_team_url, lineup, opp, history_text, singles_courts, doubles_courts)
    if provider == "ollama":
        return _run_ollama(my_team_url, usta_team_url, lineup, opp, history_text, singles_courts, doubles_courts)
    raise NotImplementedError(
        f"AI provider '{provider}' is not implemented. Supported: 'groq', 'claude', 'gemini', 'ollama'."
    )


def _player_rating_line(name: str) -> str:
    """Build a compact rating line for a player using all available sources."""
    try:
        from tennisai.modules.players.store import load_player
        pf = load_player(name)
        if not pf:
            return f"  {name}: no rating data"
        parts: list[str] = []
        if pf.wtn_singles is not None:
            parts.append(f"WTN-S={pf.wtn_singles}")
        if pf.wtn_doubles is not None:
            parts.append(f"WTN-D={pf.wtn_doubles}")
        if pf.tennisrecord_rating:
            parts.append(f"TR={pf.tennisrecord_rating}")
        elif pf.ntrp_level:
            parts.append(f"NTRP={pf.ntrp_level}")
        sr = f"{pf.singles_wins}W-{pf.singles_losses}L"
        dr = f"{pf.doubles_wins}W-{pf.doubles_losses}L"
        rating_str = "  ".join(parts) if parts else "no rating"
        return f"  {name}: {rating_str} | S={sr} D={dr}"
    except Exception:
        return f"  {name}"


def run_analysis_direct(
    lineup: dict[str, list[str]],
    opponent_lineup: dict[str, list[str]],
    history_text: str = "",
    singles_courts: int = 2,
    doubles_courts: int = 3,
    match_date: Optional[datetime.date] = None,
    opponent_name: str = "",
) -> MatchAnalysis:
    """
    Predict court outcomes without any tool calls — all data is provided inline.
    Used for backfill and any case where both lineups are already known.
    Avoids the Groq 400 error caused by malformed get_player_history XML tool calls.
    """
    # Build per-player rating lines for both teams
    all_our_players = [p for players in lineup.values() for p in players]
    all_opp_players = [p for players in opponent_lineup.values() for p in players]
    our_rating_lines = "\n".join(_player_rating_line(p) for p in all_our_players)
    opp_rating_lines = "\n".join(_player_rating_line(p) for p in all_opp_players)

    lineup_text = "\n".join(f"  {court}: {', '.join(players)}" for court, players in lineup.items())
    opp_text = "\n".join(f"  {court}: {', '.join(players)}" for court, players in opponent_lineup.items())

    singles_labels = " | ".join(f"Singles court={i}" for i in range(1, singles_courts + 1))
    doubles_labels = " | ".join(f"Doubles court={i}" for i in range(1, doubles_courts + 1))
    court_format = (
        f"MATCH FORMAT: {singles_courts} Singles + {doubles_courts} Doubles = "
        f"{singles_courts + doubles_courts} courts total. [{singles_labels} | {doubles_labels}]. "
        "Doubles courts are numbered 1, 2, 3 — NOT 3, 4, 5."
    )
    winner_note = (
        "CRITICAL — predicted_winner: write 'us' when THE CAPTAIN'S TEAM wins, "
        "'them' when THE OPPONENT wins."
    )
    date_str = str(match_date) if match_date else "unknown"
    opp_str = opponent_name or "Opponent"
    history_section = f"\n{history_text}\n" if history_text else ""

    prompt = (
        f"Predict court-by-court outcomes for our USTA match on {date_str} vs {opp_str}.\n\n"
        f"{court_format}\n\n{winner_note}\n\n"
        f"Our lineup:\n{lineup_text}\n\n"
        f"Our player ratings:\n{our_rating_lines}\n\n"
        f"Opponent lineup:\n{opp_text}\n\n"
        f"Opponent player ratings:\n{opp_rating_lines}\n"
        f"{history_section}\n"
        "Using all ratings (WTN-S for singles courts, WTN-D for doubles courts, TR as supporting context) "
        "and calibration history above, predict who wins each court with a score. "
        "Then give an overall outlook and any lineup suggestions.\n\n"
        + _OUTPUT_SCHEMA
    )

    provider = get_ai_provider()

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=get_groq_api_key())
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )
        return _parse_final_response(response.choices[0].message.content or "", "", "")

    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=get_anthropic_api_key())
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
        return _parse_final_response(final_text, "", "")

    if provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=get_gemini_api_key())
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=4096,
            ),
        )
        parts = response.candidates[0].content.parts if response.candidates else []
        final_text = next((p.text for p in parts if hasattr(p, "text") and p.text), "")
        return _parse_final_response(final_text, "", "")

    if provider == "ollama":
        from openai import OpenAI
        client = OpenAI(base_url=get_ollama_base_url(), api_key="ollama")
        response = client.chat.completions.create(
            model=get_ollama_model(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )
        return _parse_final_response(response.choices[0].message.content or "", "", "")

    raise NotImplementedError(f"AI provider '{provider}' is not implemented.")


# ---------------------------------------------------------------------------
# Response parser (shared)
# ---------------------------------------------------------------------------

def _repair_json(raw: str) -> str:
    """Fix common JSON formatting mistakes produced by small LLMs."""
    # Rule order matters: absorb stray items back into arrays BEFORE stripping parens.

    # 1. Quoted string sitting outside its array: ],("name") or ]("name") → ,"name"]
    raw = re.sub(r'\]\s*,?\s*\("([^"]*?)"\)', r', "\1"]', raw)
    # 2. Mismatched brackets wrapping a quoted string: ("name"}, {"name"), etc.
    #    Only fires when opening is ( or { — valid ["name"] is untouched.
    raw = re.sub(r'[({]\s*("(?:[^"\\]|\\.)*?")\s*[)}\]]', r'\1', raw)
    # 3. Mixed-quote keys: "key': or 'key":
    raw = re.sub(r'"([A-Za-z_][A-Za-z_0-9]*)\'\s*:', r'"\1":', raw)
    raw = re.sub(r"'([A-Za-z_][A-Za-z_0-9]*)\"?\s*:", r'"\1":', raw)
    # 4. Set literal used for lineup_suggestions: {"str", ...} → ["str", ...]
    raw = re.sub(
        r'("lineup_suggestions"\s*:\s*)\{([^{}]*)\}',
        lambda m: m.group(1) + '[' + m.group(2) + ']',
        raw,
    )
    # 5. Trailing commas before } or ]
    raw = re.sub(r',(\s*[}\]])', r'\1', raw)
    return raw


def _parse_final_response(text: str, my_team_url: str, usta_team_url: str) -> MatchAnalysis:
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        raw = json_match.group(1) if json_match else "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raw = _repair_json(raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return MatchAnalysis(
                match=Match(home_team="Unknown", away_team="Unknown"),
                my_team=Team(name="My Team", url=my_team_url),
                opponent_team=Team(name="Opponent", url=""),
                overall_outlook=text,
            )

    match_data = data.get("match", {})
    raw_date = match_data.get("date") or None
    # Reject LLM-hallucinated past dates — only accept future/today dates
    if raw_date:
        try:
            parsed_date = datetime.date.fromisoformat(str(raw_date))
            if parsed_date < datetime.date.today():
                raw_date = None
        except (ValueError, TypeError):
            raw_date = None
    match = Match(
        date=raw_date,
        home_team=match_data.get("home_team", ""),
        away_team=match_data.get("away_team", ""),
        location=match_data.get("location", ""),
    )

    def _coerce_player(p: dict) -> Player:
        rating = p.get("rating")
        if isinstance(rating, dict):
            rating = rating.get("TR") or next(
                (v for v in rating.values() if isinstance(v, (int, float))), None
            )
        return Player(
            name=p.get("name", ""),
            ntrp_level=p.get("ntrp_level") if isinstance(p.get("ntrp_level"), (int, float, type(None))) else None,
            rating=rating if isinstance(rating, (int, float, type(None))) else None,
            profile_url=p.get("profile_url", ""),
        )

    def _build_team(t: dict, fallback_url: str) -> Team:
        return Team(
            name=t.get("name", ""),
            url=t.get("url", fallback_url),
            players=[_coerce_player(p) for p in t.get("players", [])],
        )

    raw_predictions = data.get("predictions", [])
    predictions = []
    for p in raw_predictions:
        try:
            predictions.append(CourtPrediction(**p))
        except Exception:
            pass
    predictions = [_fix_predicted_winner(p) for p in predictions]

    return MatchAnalysis(
        match=match,
        my_team=_build_team(data.get("my_team", {}), my_team_url),
        opponent_team=_build_team(data.get("opponent_team", {}), ""),
        predictions=predictions,
        overall_outlook=data.get("overall_outlook", ""),
        lineup_suggestions=data.get("lineup_suggestions", []),
    )


def _fix_predicted_winner(pred: CourtPrediction) -> CourtPrediction:
    """
    Safety net: cross-check predicted_winner against the reasoning text.
    The LLM sometimes outputs 'them' for courts our team wins.
    Signals in reasoning that mean WE win: 'my team', 'our team', 'predicting a win for my team',
    'predicting a win for our team', 'i predict us', 'we will win'.
    Signals that mean THEY win: 'predicting a win for the opponent', 'opponent wins'.
    """
    r = pred.reasoning.lower()
    we_win_signals = [
        "predicting a win for my team",
        "predicting a win for our team",
        "i predict us",
        "we will win",
        "our team will win",
        "my team will win",
    ]
    they_win_signals = [
        "predicting a win for the opponent",
        "predicting a win for them",
        "opponent will win",
        "opponent wins",
    ]

    inferred_winner = None
    for s in we_win_signals:
        if s in r:
            inferred_winner = "us"
            break
    if inferred_winner is None:
        for s in they_win_signals:
            if s in r:
                inferred_winner = "them"
                break

    if inferred_winner and inferred_winner != pred.predicted_winner.lower():
        return CourtPrediction(
            court=pred.court,
            court_type=pred.court_type,
            my_players=pred.my_players,
            opponent_players=pred.opponent_players,
            predicted_winner=inferred_winner,
            predicted_score=pred.predicted_score,
            confidence=pred.confidence,
            reasoning=pred.reasoning,
        )
    return pred


# ---------------------------------------------------------------------------
# Lineup suggestion (delegates to modules/lineup/predictor)
# ---------------------------------------------------------------------------

def run_lineup_suggestion(
    my_team_url: str,
    usta_team_url: str,
    available_players: list[str],
    singles_courts: int = 2,
    doubles_courts: int = 3,
) -> dict:
    """
    Recommend an optimal lineup from available players and predict the opponent lineup.
    Returns a structured dict with our_lineup, opponent_lineup, reasoning, rotation_notes, etc.
    """
    from tennisai.modules.lineup.predictor import predict_lineup
    from tennisai.modules.players.store import rebuild_from_matches

    # Ensure player files are up to date before making recommendations
    rebuild_from_matches()

    season = get_season_context()

    # Fetch opponent via USTA
    client = USTAClient()
    upcoming, _ = client.get_schedule_and_results(usta_team_url)
    standings = client.get_standings(usta_team_url)
    client.close()

    today = datetime.date.today()
    next_match = next((m for m in upcoming if m.date and m.date >= today), None)
    opponent_name = ""
    our_team_obj = get_team_ratings(my_team_url)
    our_full_roster: set[str] = {p.name for p in our_team_obj.players}
    if next_match:
        our_name = our_team_obj.name.lower()
        opponent_name = (
            next_match.away_team
            if our_name[:8] in (next_match.home_team or "").lower()
            else next_match.home_team
        ) or ""

    # season and standings are passed through to predict_lineup

    # Fetch opponent ratings directly so LLM can predict their lineup too
    opponent_players_text = ""
    opponent_team_url = ""
    opp_full = None
    if opponent_name:
        try:
            league_teams = get_league_teams(my_team_url)
            opp_lower = opponent_name.lower()
            opp_team = next(
                (t for t in league_teams if opp_lower[:8] in t.name.lower() or t.name.lower()[:8] in opp_lower),
                None,
            )
            if opp_team:
                opponent_team_url = opp_team.url
                opp_full = get_team_ratings(opp_team.url)
                opp_lines = [
                    f"  {p.name} (NTRP {p.rating})" for p in opp_full.players
                ]
                opponent_players_text = (
                    f"\nOpponent team: {opp_full.name}\n"
                    "Opponent players:\n" + "\n".join(opp_lines)
                )
        except Exception:
            pass

    has_real_opponents = bool(opponent_players_text)

    # --- Our lineup: use the lineup module (optimizer + LLM reasoning) ---
    our_result = predict_lineup(
        available_players=available_players,
        singles_courts=singles_courts,
        doubles_courts=doubles_courts,
        season_context=season,
        standings=standings,
        opponent_name=opponent_name,
        team_label="our team",
    )

    # --- Opponent lineup: same function applied to their roster ---
    opponent_lineup: dict[str, list[str]] = {}
    if has_real_opponents and opp_full:
        opp_names = [p.name for p in opp_full.players]
        opp_result = predict_lineup(
            available_players=opp_names,
            singles_courts=singles_courts,
            doubles_courts=doubles_courts,
            opponent_name="our team",
            team_label=f"opponent ({opp_full.name})",
        )
        # Filter out any of our players that may have slipped in
        for label, players in opp_result["lineup"].items():
            clean = [p for p in players if p not in our_full_roster]
            if clean:
                opponent_lineup[label] = clean

    return {
        "our_lineup": our_result["lineup"],
        "opponent_lineup": opponent_lineup,
        "per_court_reasoning": our_result["per_court_reasoning"],
        "rotation_notes": our_result["rotation_notes"],
        "season_outlook": our_result["season_outlook"],
        "has_real_opponents": has_real_opponents,
        "opponent_name": opponent_name,
    }
