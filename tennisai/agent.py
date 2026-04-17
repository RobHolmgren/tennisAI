"""
Tennis AI agent — orchestrates data gathering and match prediction.

Default provider: Groq (free tier). Set AI_PROVIDER=claude in .env to use Claude instead.

Groq uses an OpenAI-compatible API: tools are defined with a "function" wrapper,
tool calls come back in message.tool_calls[], and results go back as role="tool" messages.
"""

import datetime
import json
import re
from typing import Any

from tennisai.config import get_ai_provider, get_groq_api_key, get_anthropic_api_key
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

def _dispatch_tool(name: str, inputs: dict[str, Any]) -> str:
    if name == "get_my_team_ratings":
        return get_team_ratings(inputs["team_url"]).model_dump_json()

    if name == "get_league_teams":
        teams = get_league_teams(inputs["team_url"])
        return json.dumps([t.model_dump() for t in teams])

    if name == "get_opponent_ratings":
        return get_team_ratings(inputs["team_url"]).model_dump_json()

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

_SYSTEM_PROMPT = (
    "You are a tennis match analyst helping a USTA adult league team captain prepare for their next match. "
    "Use the available tools to gather player ratings, schedule data, court trends, and player history, "
    "then produce a detailed court-by-court match prediction.\n\n"
    "Rating guidance: each player has two values from tennisrecord.com:\n"
    "  - ntrp_level: the official NTRP band (e.g. 3.0, 3.5) — coarse-grained\n"
    "  - rating: the estimated current rating within that band, shown to 2 decimal places (e.g. 2.98) — "
    "this is the high-priority predictor. A player with rating 3.12 is meaningfully stronger than one at 2.94 "
    "even if both are NTRP 3.0. Always use the estimated rating as the primary basis for predictions; "
    "fall back to ntrp_level only when the estimated rating is missing.\n\n"
    "Prediction factors — always consider ALL of these:\n"
    "  1. Player ratings from tennisrecord.com (primary factor)\n"
    "  2. Per-court trends: if a team consistently wins or loses a specific court position, weight that heavily\n"
    "  3. Individual player win/loss streaks and recent form (last 6 months, including matches outside this team)\n"
    "  4. Predicted opponent lineup: rank their players by rating and assign them to courts most-to-least "
    "likely based on strength ordering and any historical court assignments you can infer\n"
    "  5. Lineup optimization: review the captain's tentative lineup and suggest specific changes "
    "(player swaps, court reassignments) that would maximize the team's win probability\n"
    "  6. Team match priority: the goal is to win the overall team match (majority of courts), not to "
    "maximise individual court wins. If sacrificing a likely-loss court improves chances elsewhere, recommend it. "
    "Frame all lineup suggestions around this team-first objective.\n"
    "  7. Court difficulty weighting: Court 1 (singles or doubles) is the hardest and most valuable. "
    "Weight a win or loss at Court 1 more heavily than Court 2, and Court 2 more than Court 3. "
    "Prioritise fielding stronger players at lower-numbered courts.\n\n"
    "When you have all the data you need, provide a final analysis — do not ask follow-up questions."
)

_OUTPUT_SCHEMA = (
    "Return your final analysis as a JSON object:\n"
    "{\n"
    '  "match": {"date": "YYYY-MM-DD", "home_team": "...", "away_team": "...", "location": "..."},\n'
    '  "my_team": {"name": "...", "url": "...", "players": [{"name": "...", "ntrp_level": 3.0, "rating": 2.94}]},\n'
    '  "opponent_team": {"name": "...", "url": "...", "players": [{"name": "...", "ntrp_level": 3.0, "rating": 2.98}]},\n'
    '  "my_recent_results": [{"date": "YYYY-MM-DD", "opponent": "...", "score": "3-2", "won": true}],\n'
    '  "opponent_recent_results": [{"date": "YYYY-MM-DD", "opponent": "...", "score": "1-4", "won": false}],\n'
    '  "predictions": [\n'
    '    {"court": 1, "court_type": "Singles", "my_players": ["..."], "opponent_players": ["..."],\n'
    '     "predicted_winner": "us", "confidence": "high",\n'
    '     "reasoning": "... include rating comparison, court trend, and player form factors ..."}\n'
    "  ],\n"
    '  "overall_outlook": "... include team form, momentum, key match-ups ...",\n'
    '  "lineup_suggestions": [\n'
    '    "Consider moving Player A from Court 2 Singles to Court 1 Singles — higher rating (3.12 vs 2.94).",\n'
    '    "Pair Player B and Player C at Doubles 1 — combined rating is highest vs weakest opponent court."\n'
    "  ]\n"
    "}"
)


def _build_user_message(my_team_url: str, usta_team_url: str, lineup: dict[str, list[str]]) -> str:
    lineup_text = "\n".join(f"  {court}: {', '.join(players)}" for court, players in lineup.items())
    return (
        f"Please analyze our next match and predict the outcome for each court.\n\n"
        f"Our team's tennisrecord.com URL: {my_team_url}\n"
        f"Our USTA TennisLink URL: {usta_team_url}\n\n"
        f"Our tentative lineup:\n{lineup_text}\n\n"
        "Steps to follow:\n"
        "1. Fetch our team's player ratings from tennisrecord.com (note profile_url for each player)\n"
        "2. Call get_usta_data to get our next match date/opponent AND our recent match results\n"
        "3. Call get_court_trends to get our per-court win/loss history from match scorecards\n"
        "4. Find our league teams and identify the opponent's tennisrecord.com URL from the next match\n"
        "5. Fetch the opponent's player ratings\n"
        "6. For each player in our lineup who has a profile_url, call get_player_history to get their "
        "recent form over the last 6 months\n"
        "7. Predict the opponent's most likely lineup by ranking their players strongest-to-weakest "
        "and assigning them to courts accordingly\n"
        "8. Compare our lineup against the predicted opponent lineup, factoring in ratings, "
        "per-court trends, and individual player form — predict each court result\n"
        "9. Review the tentative lineup and suggest specific changes to maximize our win chances\n"
        "10. Provide an overall match outlook\n\n"
        + _OUTPUT_SCHEMA
    )


# ---------------------------------------------------------------------------
# Groq agent loop
# ---------------------------------------------------------------------------

def _run_groq(my_team_url: str, usta_team_url: str, lineup: dict[str, list[str]]) -> MatchAnalysis:
    from groq import Groq

    client = Groq(api_key=get_groq_api_key())
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(my_team_url, usta_team_url, lineup)},
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

def _run_claude(my_team_url: str, usta_team_url: str, lineup: dict[str, list[str]]) -> MatchAnalysis:
    import anthropic

    client = anthropic.Anthropic(api_key=get_anthropic_api_key())
    messages: list[dict] = [
        {"role": "user", "content": _build_user_message(my_team_url, usta_team_url, lineup)},
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
) -> MatchAnalysis:
    provider = get_ai_provider()
    if provider == "groq":
        return _run_groq(my_team_url, usta_team_url, lineup)
    if provider == "claude":
        return _run_claude(my_team_url, usta_team_url, lineup)
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

    return MatchAnalysis(
        match=match,
        my_team=_build_team(data.get("my_team", {}), my_team_url),
        opponent_team=_build_team(data.get("opponent_team", {}), ""),
        predictions=[CourtPrediction(**p) for p in data.get("predictions", [])],
        overall_outlook=data.get("overall_outlook", ""),
        lineup_suggestions=data.get("lineup_suggestions", []),
    )
