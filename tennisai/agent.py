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

from tennisai.config import get_ai_provider, get_groq_api_key, get_anthropic_api_key, get_scoring_format
from tennisai.models import CourtPrediction, Match, MatchAnalysis, Player, Team
from tennisai.tools.tennisrecord import get_league_teams, get_player_history, get_team_ratings
from tennisai.tools.usta import USTAClient

GROQ_MODEL = "llama-3.3-70b-versatile"
CLAUDE_MODEL = "claude-sonnet-4-6"
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
        "Ratings: use 'rating' (2-decimal, e.g. 2.98) as primary predictor; fall back to ntrp_level only if missing. "
        "3.12 beats 2.94 even if both are NTRP 3.0.\n\n"
        "Weigh these factors: (1) ratings, (2) per-court win/loss trends, (3) player form last 6 months, "
        "(4) predicted opponent lineup strongest-to-weakest by court, (5) lineup changes to maximise team wins. "
        "Goal is team match victory (most courts), not individual courts. "
        "Court 1 > Court 2 > Court 3 in difficulty and importance — put stronger players lower-numbered.\n\n"
        "Provide final analysis without asking follow-up questions."
    )


_SYSTEM_PROMPT = _build_system_prompt()

_OUTPUT_SCHEMA = (
    "Return JSON only — no prose outside the JSON block:\n"
    "{\n"
    '  "match": {"date": "YYYY-MM-DD", "home_team": "", "away_team": "", "location": ""},\n'
    '  "my_team": {"name": "", "url": "", "players": [{"name": "", "ntrp_level": 3.0, "rating": 2.94}]},\n'
    '  "opponent_team": {"name": "", "url": "", "players": [{"name": "", "ntrp_level": 3.0, "rating": 2.98}]},\n'
    '  "predictions": [\n'
    '    {"court": 1, "court_type": "Singles", "my_players": [""], "opponent_players": [""],\n'
    '     "predicted_winner": "us", "predicted_score": "6-3 6-2", "confidence": "high", "reasoning": ""}\n'
    "  ],\n"
    '  "overall_outlook": "",\n'
    '  "lineup_suggestions": [""]\n'
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
    raise NotImplementedError(
        f"AI provider '{provider}' is not implemented. Supported: 'groq', 'claude'."
    )


# ---------------------------------------------------------------------------
# Response parser (shared)
# ---------------------------------------------------------------------------

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
        return MatchAnalysis(
            match=Match(home_team="Unknown", away_team="Unknown"),
            my_team=Team(name="My Team", url=my_team_url),
            opponent_team=Team(name="Opponent", url=""),
            overall_outlook=text,
        )

    match_data = data.get("match", {})
    match = Match(
        date=match_data.get("date"),
        home_team=match_data.get("home_team", ""),
        away_team=match_data.get("away_team", ""),
        location=match_data.get("location", ""),
    )

    def _build_team(t: dict, fallback_url: str) -> Team:
        return Team(
            name=t.get("name", ""),
            url=t.get("url", fallback_url),
            players=[Player(**p) for p in t.get("players", [])],
        )

    predictions = [CourtPrediction(**p) for p in data.get("predictions", [])]
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
