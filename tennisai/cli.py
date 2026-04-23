"""
Tennis AI CLI — entry point.

Usage (URLs configured in .env):
    python -m tennisai analyze
    python -m tennisai analyze --output-csv results.csv

Usage (URLs as arguments):
    python -m tennisai analyze --team-url <url> --usta-url <url>
"""

import csv
import datetime
import json
import sys
from typing import Optional

import click

from tennisai.config import (
    get_singles_courts_override, get_my_team_url, get_usta_team_url,
    get_all_teams, get_active_team_index, set_active_team,
)
from tennisai.models import CourtResult, MatchAnalysis
from tennisai.modules.matches import load_all_matches, load_match
from tennisai.modules.results import predict_match_results, apply_learnings
from tennisai.tools.history import format_history_for_prompt, get_recent_records, record_prediction, update_result


def _resolve_team_url(option_value: Optional[str]) -> str:
    if option_value:
        return option_value
    return get_my_team_url()


def _resolve_usta_url(option_value: Optional[str]) -> str:
    if option_value:
        return option_value
    return get_usta_team_url()


def _enrich_opponent_players(players, our_player_names: set) -> None:
    """
    Create/update player files for opponent players with tennisrecord ratings and WTN.
    Skips players already on our roster. Only fetches WTN for players missing it.
    """
    from tennisai.modules.players.models import PlayerFile
    from tennisai.modules.players.store import load_player, save_player
    from tennisai.tools.usta_wtn import fetch_wtn_batch

    opp_players = [p for p in players if p.name not in our_player_names]
    if not opp_players:
        return

    # Save tennisrecord ratings first so player files exist before WTN fetch
    for p in opp_players:
        pf = load_player(p.name) or PlayerFile(name=p.name)
        pf.tennisrecord_rating = p.rating or pf.tennisrecord_rating
        pf.ntrp_level = p.ntrp_level or pf.ntrp_level
        pf.profile_url = p.profile_url or pf.profile_url
        save_player(pf)

    # Fetch WTN only for players who don't already have it
    missing = [
        p.name for p in opp_players
        if (pf := load_player(p.name)) is None
        or (pf.wtn_singles is None and pf.wtn_doubles is None)
    ]
    if missing:
        click.echo(f"Fetching WTN ratings for {len(missing)} opponent player(s)...")
        wtn_results = fetch_wtn_batch(missing)
        for name, wtn in wtn_results.items():
            pf = load_player(name)
            if not pf:
                continue
            if wtn.get("singles") is not None:
                pf.wtn_singles = wtn["singles"]
            if wtn.get("doubles") is not None:
                pf.wtn_doubles = wtn["doubles"]
            save_player(pf)


def _pick_match(usta_url: str) -> dict:
    """
    Let the user pick a match: upcoming matches + past matches within 3 months.
    Returns a dict with keys: date, home_team, away_team, location, opponent_name, is_upcoming.
    """
    from tennisai.config import get_my_team_name
    from tennisai.tools.usta import USTAClient

    click.echo("\nFetching match schedule from USTA TennisLink...")
    client = USTAClient()
    upcoming, results = client.get_schedule_and_results(usta_url)
    client.close()

    today = datetime.date.today()
    three_months_ago = today - datetime.timedelta(days=90)
    our_name_lower = get_my_team_name().lower()

    # Build a unified list: upcoming first (soonest first), then past (most recent first)
    choices: list[dict] = []

    for m in sorted(upcoming, key=lambda x: x.date or today):
        if m.date and m.date >= today:
            home_lower = (m.home_team or "").lower()
            opp = (m.away_team if our_name_lower[:8] in home_lower else m.home_team) or "Unknown"
            choices.append({
                "date": m.date,
                "home_team": m.home_team or "",
                "away_team": m.away_team or "",
                "location": m.location or "",
                "opponent_name": opp,
                "is_upcoming": True,
            })

    # Past matches from USTA
    past: list[dict] = []
    seen_dates: set[str] = set()
    for r in results:
        if r.date and three_months_ago <= r.date < today:
            key = f"{r.date}-{r.opponent}"
            if key not in seen_dates:
                seen_dates.add(key)
                past.append({
                    "date": r.date,
                    "home_team": "",
                    "away_team": "",
                    "location": r.location or "",
                    "opponent_name": r.opponent or "Unknown",
                    "is_upcoming": False,
                })

    # Also include locally saved match files not already covered
    for mf in load_all_matches():
        if mf.match_date and three_months_ago <= mf.match_date < today:
            key = f"{mf.match_date}-{mf.opponent}"
            if key not in seen_dates:
                seen_dates.add(key)
                past.append({
                    "date": mf.match_date,
                    "home_team": mf.home_team,
                    "away_team": mf.away_team,
                    "location": mf.location,
                    "opponent_name": mf.opponent or "Unknown",
                    "is_upcoming": False,
                    "match_id": mf.id,
                })

    past.sort(key=lambda x: x["date"], reverse=True)
    choices.extend(past)

    if not choices:
        click.echo("No matches found in USTA schedule (upcoming or last 3 months).", err=True)
        sys.exit(1)

    click.echo("\n--- Select a match ---")
    for i, m in enumerate(choices, 1):
        tag = "UPCOMING" if m["is_upcoming"] else "past   "
        loc = f"  @ {m['location']}" if m["location"] else ""
        click.echo(f"  {i:>2}. [{tag}] {m['date']}  vs {m['opponent_name']}{loc}")

    while True:
        raw = click.prompt("\nSelect match number").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        click.echo(f"  Enter a number between 1 and {len(choices)}.")


def _build_court_slots(singles: int, doubles: int) -> list[tuple[str, int]]:
    """Build the list of (court_label, player_count) tuples for the given match format."""
    slots = []
    for i in range(1, singles + 1):
        slots.append((f"Court {i} Singles", 1))
    for i in range(1, doubles + 1):
        slots.append((f"Court {i} Doubles", 2))
    return slots


def _collect_lineup(roster: list[str], court_slots: list[tuple[str, int]]) -> dict[str, list[str]]:
    """
    Prompt the captain to build their lineup by selecting player numbers from the roster.
    The roster is displayed once upfront; the captain types numbers for each court slot.
    """
    click.echo("\n--- Your team roster ---")
    for i, name in enumerate(roster, 1):
        click.echo(f"  {i:>2}. {name}")

    click.echo("\n--- Enter your tentative lineup ---")
    click.echo("Type player number(s) or press Enter to skip a court.\n")

    lineup: dict[str, list[str]] = {}

    for court_label, player_count in court_slots:
        players: list[str] = []
        for i in range(1, player_count + 1):
            label = f"{court_label} - Player {i}" if player_count > 1 else court_label
            while True:
                raw = click.prompt(f"  {label}", default="", show_default=False).strip()
                if not raw:
                    break
                if raw.isdigit() and 1 <= int(raw) <= len(roster):
                    players.append(roster[int(raw) - 1])
                    break
                # Accept free-text name too (in case roster is incomplete)
                if not raw.isdigit():
                    players.append(raw)
                    break
                click.echo(f"  Please enter a number between 1 and {len(roster)}, or a name.")
        if players:
            lineup[court_label] = players

    return lineup


def _collect_opponent_lineup(
    opponent_roster: list[str],
    court_slots: list[tuple[str, int]],
) -> dict[str, list[str]]:
    """
    Optionally collect the opponent's known lineup.
    Returns an empty dict if the captain skips this step.
    """
    click.echo("\n--- Opponent lineup (optional) ---")
    click.echo("If you know who they're playing, enter it now to improve predictions.")
    if not click.confirm("  Do you know the opponent's lineup?", default=False):
        return {}

    if opponent_roster:
        click.echo("\n  Opponent roster:")
        for i, name in enumerate(opponent_roster, 1):
            click.echo(f"  {i:>2}. {name}")
        click.echo()

    lineup: dict[str, list[str]] = {}
    for court_label, player_count in court_slots:
        players: list[str] = []
        for i in range(1, player_count + 1):
            label = f"{court_label} - Player {i}" if player_count > 1 else court_label
            while True:
                raw = click.prompt(f"  {label}", default="", show_default=False).strip()
                if not raw:
                    break
                if opponent_roster and raw.isdigit() and 1 <= int(raw) <= len(opponent_roster):
                    players.append(opponent_roster[int(raw) - 1])
                    break
                if not raw.isdigit():
                    players.append(raw)
                    break
                click.echo(f"  Enter a number 1-{len(opponent_roster)}, a name, or press Enter to skip.")
        if players:
            lineup[court_label] = players

    return lineup


def _print_analysis(analysis: MatchAnalysis) -> None:
    """Print a formatted match analysis to stdout."""
    m = analysis.match
    click.echo("\n" + "=" * 60)
    click.echo("MATCH ANALYSIS")
    click.echo("=" * 60)
    if m.date:
        click.echo(f"Date:     {m.date}")
    if m.home_team:
        click.echo(f"Home:     {m.home_team}")
    if m.away_team:
        click.echo(f"Away:     {m.away_team}")
    if m.location:
        click.echo(f"Location: {m.location}")

    if analysis.predictions:
        click.echo("\n--- Court Predictions ---")
        for pred in analysis.predictions:
            players_us = ", ".join(pred.my_players) or "TBD"
            players_them = ", ".join(pred.opponent_players) or "TBD"
            winner_label = f"Us ({players_us})" if pred.predicted_winner.lower() == "us" else f"Them ({players_them})"
            score_str = f"  {pred.predicted_score}" if pred.predicted_score else ""
            click.echo(
                f"\nCourt {pred.court} {pred.court_type}"
                f"\n  Us:               {players_us}"
                f"\n  Them:             {players_them}"
                f"\n  Predicted winner: {winner_label}{score_str} ({pred.confidence} confidence)"
                f"\n  Why:              {pred.reasoning}"
            )

    if analysis.overall_outlook:
        click.echo("\n--- Overall Outlook ---")
        click.echo(analysis.overall_outlook)

    if analysis.lineup_suggestions:
        click.echo("\n--- Lineup Suggestions ---")
        for i, suggestion in enumerate(analysis.lineup_suggestions, 1):
            click.echo(f"  {i}. {suggestion}")

    click.echo("=" * 60 + "\n")


def _write_csv(analysis: MatchAnalysis, path: str) -> None:
    """Write court predictions and lineup suggestions to a CSV file."""
    pred_path = path
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "court", "court_type", "my_players", "opponent_players",
            "predicted_winner", "predicted_score", "confidence", "reasoning",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in analysis.predictions:
            writer.writerow({
                "court": pred.court,
                "court_type": pred.court_type,
                "my_players": "; ".join(pred.my_players),
                "opponent_players": "; ".join(pred.opponent_players),
                "predicted_winner": pred.predicted_winner,
                "predicted_score": pred.predicted_score,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
            })
        # Append lineup suggestions as a separate section
        if analysis.lineup_suggestions:
            writer.writerow({})  # blank row
            for i, suggestion in enumerate(analysis.lineup_suggestions, 1):
                writer.writerow({"court": f"Suggestion {i}", "reasoning": suggestion})
    click.echo(f"Predictions saved to {pred_path}")


@click.group()
def cli() -> None:
    """TennisAI — AI-powered USTA match analysis."""


@cli.command()
@click.option(
    "--team-url",
    default=None,
    help="Your team's tennisrecord.com profile URL (defaults to MY_TEAM_URL in .env)",
)
@click.option(
    "--usta-url",
    default=None,
    help="Your team's tennislink.usta.com Stats & Standings URL (defaults to USTA_TEAM_URL in .env)",
)
@click.option(
    "--output-csv",
    default=None,
    help="Optional path to save court predictions as a CSV file",
)
@click.option(
    "--team",
    "team_num",
    type=int,
    default=None,
    help="Team number to use for this run (overrides ACTIVE_TEAM; does not persist)",
)
def analyze(
    team_url: Optional[str],
    usta_url: Optional[str],
    output_csv: Optional[str],
    team_num: Optional[int],
) -> None:
    """Analyze a USTA match and predict court outcomes."""
    try:
        import os
        if team_num is not None:
            os.environ["ACTIVE_TEAM"] = str(team_num)

        from tennisai.config import get_my_team_name
        click.echo(f"\nTeam: {get_my_team_name()}")

        team_url = _resolve_team_url(team_url)
        usta_url = _resolve_usta_url(usta_url)

        from tennisai.tools.tennisrecord import get_league_teams, get_team_ratings
        from tennisai.tools.usta import USTAClient

        # --- Step 1: Pick match ---
        selected = _pick_match(usta_url)
        opp_name = selected["opponent_name"]
        click.echo(
            f"\n  Selected: {selected['date']}  vs {opp_name}"
            + (f"  @ {selected['location']}" if selected["location"] else "")
        )

        # --- Step 2: Fetch roster ---
        click.echo("\nFetching team roster from tennisrecord.com...")
        try:
            team = get_team_ratings(team_url)
            roster = [p.name for p in team.players]
        except Exception:
            roster = []
        if not roster:
            click.echo("Could not fetch roster — you can still type player names directly.\n")

        # --- Step 3: Match format ---
        singles_override = get_singles_courts_override()
        if singles_override is not None:
            singles, doubles = singles_override, 3
            click.echo(f"  Match format: {singles} Singles, {doubles} Doubles (from .env)")
        else:
            click.echo("Fetching match format from USTA TennisLink...")
            try:
                usta_client = USTAClient()
                match_format = usta_client.get_match_format(usta_url)
                usta_client.close()
                singles = match_format.singles_courts
                doubles = match_format.doubles_courts
                click.echo(f"  Match format: {singles} Singles, {doubles} Doubles")
            except Exception:
                singles, doubles = 2, 3
                click.echo(f"  Could not detect format — defaulting to {singles} Singles, {doubles} Doubles.")

        court_slots = _build_court_slots(singles, doubles)

        # --- Step 4: Our lineup ---
        lineup = _collect_lineup(roster, court_slots)
        if not lineup:
            click.echo("No lineup entered. At least one court must have a player assigned.", err=True)
            sys.exit(1)

        # --- Step 5: Opponent roster (best-effort) ---
        opponent_roster: list[str] = []
        try:
            our_full = get_team_ratings(team_url)
            our_player_names = {p.name for p in our_full.players}
            league_teams = get_league_teams(team_url)
            opp_team_obj = next(
                (t for t in league_teams
                 if opp_name.lower()[:8] in t.name.lower() and t.url != team_url),
                None,
            )
            if opp_team_obj:
                opp_team = get_team_ratings(opp_team_obj.url)
                opponent_roster = [p.name for p in opp_team.players
                                   if p.name not in our_player_names]
                _enrich_opponent_players(opp_team.players, our_player_names)
        except Exception:
            pass

        opponent_lineup = _collect_opponent_lineup(opponent_roster, court_slots)

        # When opponent lineup is unknown but we have their roster, predict it using
        # the optimizer + LLM so the reliable direct analysis path can be used.
        if not opponent_lineup and opponent_roster:
            click.echo("\nPredicting opponent lineup from their roster...")
            from tennisai.modules.lineup.predictor import predict_lineup
            opp_result = predict_lineup(
                available_players=opponent_roster,
                singles_courts=singles,
                doubles_courts=doubles,
                opponent_name=opp_name,
                team_label=f"opponent ({opp_name})",
            )
            opponent_lineup = opp_result["lineup"]
            click.echo("\n--- Predicted opponent lineup ---")
            for label, players in opponent_lineup.items():
                click.echo(f"  {label}: {', '.join(players) if players else 'TBD'}")

        history_text = format_history_for_prompt(get_recent_records(3))
        click.echo("\nAnalyzing match... (this may take a moment)")
        analysis = predict_match_results(
            my_team_url=team_url,
            usta_team_url=usta_url,
            lineup=lineup,
            opponent_lineup=opponent_lineup,
            history_text=history_text,
            singles_courts=singles,
            doubles_courts=doubles,
        )

        _print_analysis(analysis)

        match_record = record_prediction(analysis, lineup, opponent_lineup)
        click.echo(f"Prediction saved (ID: {match_record.id}). Use 'record-result --id {match_record.id}' after the match.")

        if output_csv:
            _write_csv(analysis, output_csv)

    except EnvironmentError as exc:
        click.echo(f"\nConfiguration error: {exc}", err=True)
        sys.exit(1)
    except PermissionError as exc:
        click.echo(f"\nAuthentication error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--team-url",
    default=None,
    help="Your team's tennisrecord.com profile URL (defaults to MY_TEAM_URL in .env)",
)
def check_tennisrecord(team_url: Optional[str]) -> None:
    """Test tennisrecord.com scraping: fetch your team's players and league teams."""
    team_url = _resolve_team_url(team_url)
    from tennisai.tools.tennisrecord import get_team_ratings, get_league_teams

    click.echo("\n[1/2] Fetching your team's player ratings...")
    try:
        team = get_team_ratings(team_url)
        click.echo(f"  Team: {team.name}")
        if team.players:
            click.echo(f"  Players found ({len(team.players)}):")
            for p in team.players:
                ntrp_str = f"NTRP {p.ntrp_level}" if p.ntrp_level is not None else "NTRP ?"
                rating_str = f"  Rating {p.rating:.2f}" if p.rating is not None else "  Rating ?"
                click.echo(f"    - {p.name:<30} {ntrp_str}  {rating_str}")
        else:
            click.echo("  WARNING: No players found. The page scraper may need adjustment.")
            click.echo("  Tip: Run with --debug to see the raw page structure.")
    except Exception as exc:
        click.echo(f"  ERROR: {exc}", err=True)
        sys.exit(1)

    click.echo("\n[2/2] Fetching league teams...")
    try:
        teams = get_league_teams(team_url)
        if teams:
            click.echo(f"  Teams in your league ({len(teams)}):")
            for t in teams:
                click.echo(f"    - {t.name}")
                click.echo(f"      {t.url}")
        else:
            click.echo("  WARNING: No league teams found.")
    except Exception as exc:
        click.echo(f"  ERROR: {exc}", err=True)
        sys.exit(1)

    click.echo("\ntennisrecord.com check complete.")


@cli.command()
@click.option(
    "--usta-url",
    default=None,
    help="Your team's tennislink.usta.com Stats & Standings URL (defaults to USTA_TEAM_URL in .env)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Print raw page content to help identify the correct schedule endpoint",
)
def check_usta(usta_url: Optional[str], debug: bool) -> None:
    """Test USTA TennisLink login and fetch your upcoming match schedule."""
    usta_url = _resolve_usta_url(usta_url)
    from tennisai.tools.usta import USTAClient

    click.echo("\n[1/2] Logging in to USTA TennisLink...")
    try:
        client = USTAClient()
        client.login()
        click.echo("  Login successful.")
    except EnvironmentError as exc:
        click.echo(f"  Configuration error: {exc}", err=True)
        sys.exit(1)
    except PermissionError as exc:
        click.echo(f"  Login failed: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"  ERROR: {exc}", err=True)
        sys.exit(1)

    if debug:
        click.echo("\n[DEBUG] Fetching raw page content after login...")
        client.debug_page(usta_url)
        return

    click.echo("\n[2/2] Fetching match schedule...")
    try:
        matches = client.get_schedule(usta_url)
        if matches:
            click.echo(f"  Matches found ({len(matches)}):")
            for m in matches:
                date_str = str(m.date) if m.date else "Unknown date"
                click.echo(f"    - {date_str}: {m.home_team} vs {m.away_team}")
                if m.location:
                    click.echo(f"      @ {m.location}")
        else:
            click.echo("  WARNING: No matches found in the schedule.")

        next_match = client.get_next_match(usta_url)
        if next_match:
            click.echo(f"\n  Next match: {next_match.date} — {next_match.home_team} vs {next_match.away_team}")
        else:
            click.echo("\n  No upcoming matches found.")
    except Exception as exc:
        click.echo(f"  ERROR: {exc}", err=True)
        sys.exit(1)

    click.echo("\nUSTA TennisLink check complete.")


@cli.command("suggest-lineup")
@click.option("--team-url", default=None, help="tennisrecord.com team URL (defaults to MY_TEAM_URL in .env)")
@click.option("--usta-url", default=None, help="USTA TennisLink URL (defaults to USTA_TEAM_URL in .env)")
@click.option("--team", "team_num", type=int, default=None,
              help="Team number to use for this run (overrides ACTIVE_TEAM; does not persist)")
def suggest_lineup(team_url: Optional[str], usta_url: Optional[str], team_num: Optional[int]) -> None:
    """Predict opponent lineup, optionally suggest ours, then run match analysis."""
    try:
        import os
        if team_num is not None:
            os.environ["ACTIVE_TEAM"] = str(team_num)

        from tennisai.config import get_my_team_name
        click.echo(f"\nTeam: {get_my_team_name()}")

        team_url = _resolve_team_url(team_url)
        usta_url = _resolve_usta_url(usta_url)

        from tennisai.agent import run_lineup_suggestion
        from tennisai.config import get_singles_courts_override
        from tennisai.tools.history import format_history_for_prompt, get_recent_records, record_prediction
        from tennisai.tools.tennisrecord import get_team_ratings

        singles = get_singles_courts_override() or 2
        doubles = 3
        court_slots = _build_court_slots(singles, doubles)

        # --- Step 1: Pick match ---
        selected = _pick_match(usta_url)
        opp_name = selected["opponent_name"]
        click.echo(
            f"\n  Selected: {selected['date']}  vs {opp_name}"
            + (f"  @ {selected['location']}" if selected["location"] else "")
        )

        # --- Step 2: Fetch our roster ---
        click.echo("\nFetching roster from tennisrecord.com...")
        try:
            team = get_team_ratings(team_url)
            roster = [p.name for p in team.players]
        except Exception:
            roster = []

        # Enrich opponent player files with TR ratings + WTN before analysis
        try:
            from tennisai.tools.tennisrecord import get_league_teams
            our_player_names: set[str] = set(roster)
            league_teams = get_league_teams(team_url)
            opp_team_obj = next(
                (t for t in league_teams
                 if opp_name.lower()[:8] in t.name.lower() and t.url != team_url),
                None,
            )
            if opp_team_obj:
                opp_team_data = get_team_ratings(opp_team_obj.url)
                _enrich_opponent_players(opp_team_data.players, our_player_names)
        except Exception:
            pass

        click.echo("\n--- Select available players for this match ---")
        if roster:
            for i, name in enumerate(roster, 1):
                click.echo(f"  {i:>2}. {name}")
            click.echo("\nEnter player numbers separated by spaces (e.g. 1 3 5 7 9), or names:")
        raw = click.prompt("Available players").strip()
        available: list[str] = []
        for token in raw.split():
            if token.isdigit() and roster and 1 <= int(token) <= len(roster):
                available.append(roster[int(token) - 1])
            elif not token.isdigit():
                available.append(token)
        if not available:
            click.echo("No players entered.", err=True)
            sys.exit(1)
        click.echo(f"\n  {len(available)} players selected: {', '.join(available)}")

        # --- Step 2: Predict opponent lineup + optionally our lineup ---
        click.echo("\nFetching opponent data and generating lineup recommendation... (this may take a moment)")
        suggestion = run_lineup_suggestion(
            my_team_url=team_url,
            usta_team_url=usta_url,
            available_players=available,
            singles_courts=singles,
            doubles_courts=doubles,
        )

        has_real_opponents = suggestion["has_real_opponents"]
        ai_our_lineup: dict[str, list[str]] = suggestion["our_lineup"]
        opponent_lineup: dict[str, list[str]] = suggestion["opponent_lineup"]
        opponent_name = suggestion.get("opponent_name", "Opponent")

        # --- Step 3: Show opponent lineup ---
        click.echo("\n" + "=" * 60)
        click.echo(f"OPPONENT LINEUP PREDICTION: {opponent_name or 'Opponent'}")
        click.echo("=" * 60)
        if has_real_opponents and opponent_lineup:
            for label, players in opponent_lineup.items():
                click.echo(f"  {label}: {', '.join(players)}")
        else:
            click.echo("  Opponent roster not found on tennisrecord.com — predictions will use ratings only.")

        # --- Step 4: Ask user: AI lineup or manual? ---
        click.echo("\n" + "=" * 60)
        use_ai = click.confirm("Use TennisAI's suggested lineup for our team?", default=True)

        if use_ai:
            our_lineup = ai_our_lineup
            click.echo("\n--- AI Suggested Lineup ---")
            per_court = suggestion.get("per_court_reasoning", {})
            for label, players in our_lineup.items():
                reason = per_court.get(label, "")
                click.echo(f"  {label}: {', '.join(players) or 'TBD'}" + (f"  ({reason})" if reason else ""))
            if suggestion.get("rotation_notes"):
                click.echo("\nRotation notes:")
                for note in suggestion["rotation_notes"]:
                    click.echo(f"  • {note}")
        else:
            click.echo("\n--- Enter Your Lineup ---")
            our_lineup = _collect_lineup(roster, court_slots)

        # --- Step 5: Run match analysis once ---
        click.echo("\nRunning match analysis...")
        history_text = format_history_for_prompt(get_recent_records(3))
        analysis = predict_match_results(
            my_team_url=team_url,
            usta_team_url=usta_url,
            lineup=our_lineup,
            opponent_lineup=opponent_lineup if has_real_opponents else None,
            history_text=history_text,
            singles_courts=singles,
            doubles_courts=doubles,
        )
        _print_analysis(analysis)
        record_prediction(analysis, our_lineup, opponent_lineup)

    except EnvironmentError as exc:
        click.echo(f"\nConfiguration error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@cli.command("record-result")
@click.option("--id", "record_id", default=None, help="Match record ID printed after analyze (e.g. a3f2b1c0)")
def record_result(record_id: Optional[str]) -> None:
    """Enter the actual court results after a match to improve future predictions."""
    from tennisai.tools.history import load_history

    records = load_history()
    if not records:
        click.echo("No match history found. Run 'analyze' first.")
        return

    # Pick the record to update
    if not record_id:
        click.echo("\nRecent predictions (newest first):")
        for r in reversed(records[-8:]):
            date_str = str(r.match_date) if r.match_date else r.recorded_at[:10]
            status = "✓ results recorded" if r.actual_results else "pending"
            click.echo(f"  {r.id}  {date_str}  vs {r.opponent}  [{status}]")
        record_id = click.prompt("\nEnter record ID to update").strip()

    record = next((r for r in records if r.id == record_id), None)
    if not record:
        click.echo(f"No record found with ID '{record_id}'.", err=True)
        return

    click.echo(f"\nEntering results for: {record.match_date} vs {record.opponent}")
    click.echo("For each court enter 'us' or 'them' (or press Enter to skip).\n")

    actual_results: list[CourtResult] = []
    for pred in record.predictions:
        our = ", ".join(pred.my_players) or "TBD"
        them = ", ".join(pred.opponent_players) or "TBD"
        click.echo(f"  Court {pred.court} {pred.court_type}: {our} vs {them} (predicted: {pred.predicted_winner})")
        while True:
            raw = click.prompt("  Winner", default="", show_default=False).strip().lower()
            if raw == "":
                break
            if raw in ("us", "them"):
                score = click.prompt("  Score (optional, e.g. 6-3 6-4)", default="", show_default=False).strip()
                actual_results.append(CourtResult(
                    court=pred.court,
                    court_type=pred.court_type,
                    winner=raw,
                    score=score,
                ))
                break
            click.echo("  Please enter 'us' or 'them'.")

    if not actual_results:
        click.echo("No results entered.")
        return

    updated = update_result(record_id, actual_results)
    if updated:
        correct = sum(
            1 for a in updated.actual_results
            for p in updated.predictions
            if p.court == a.court and p.court_type == a.court_type
            and p.predicted_winner.lower() == a.winner.lower()
        )
        total = len(updated.actual_results)
        click.echo(f"\nResults saved. Prediction accuracy: {correct}/{total} courts correct.")
        click.echo(f"Overall match result: {updated.overall_actual}.")

        # Apply learnings: update player calibration files
        match_file = load_match(record_id)
        if match_file:
            summary = apply_learnings(match_file)
            if summary:
                click.echo("\nPlayer calibration updated:")
                for line in summary:
                    click.echo(line)


@cli.command("list-matches")
def list_matches() -> None:
    """List all saved matches (upcoming and completed)."""
    matches = load_all_matches()
    if not matches:
        click.echo("No matches saved yet. Run 'analyze' or 'suggest-lineup' to create one.")
        return

    click.echo(f"\n{'ID':<10} {'Date':<12} {'Opponent':<30} {'Predicted':<10} {'Actual':<10} Status")
    click.echo("-" * 85)
    for m in reversed(matches):
        date_str = str(m.match_date) if m.match_date else "TBD"
        actual = m.overall_actual or "-"
        click.echo(
            f"{m.id:<10} {date_str:<12} {m.opponent:<30} {m.overall_predicted:<10} {actual:<10} {m.status}"
        )
    click.echo()


@cli.command("view-match")
@click.argument("match_id")
def view_match(match_id: str) -> None:
    """Show full detail for a saved match."""
    m = load_match(match_id)
    if not m:
        click.echo(f"No match found with ID '{match_id}'.", err=True)
        return

    click.echo("\n" + "=" * 60)
    click.echo(f"MATCH  {m.id}")
    click.echo("=" * 60)
    click.echo(f"Date:      {m.match_date or 'TBD'}")
    click.echo(f"Opponent:  {m.opponent or 'Unknown'}")
    click.echo(f"Location:  {m.location or 'Unknown'}")
    click.echo(f"Status:    {m.status}")
    click.echo(f"Predicted: {m.overall_predicted}   Actual: {m.overall_actual or 'not yet recorded'}")

    if m.our_lineup:
        click.echo("\n--- Our Lineup ---")
        for court, players in m.our_lineup.items():
            click.echo(f"  {court}: {', '.join(players)}")

    if m.opponent_lineup:
        click.echo("\n--- Opponent Lineup ---")
        for court, players in m.opponent_lineup.items():
            click.echo(f"  {court}: {', '.join(players)}")

    if m.predictions:
        click.echo("\n--- Court Predictions ---")
        for pred in m.predictions:
            actual = next(
                (a for a in m.actual_results
                 if a.court == pred.court and a.court_type == pred.court_type),
                None,
            )
            ok = "✓" if actual and actual.winner.lower() == pred.predicted_winner.lower() else (
                "✗" if actual else "?"
            )
            actual_str = f"  actual={actual.winner}" if actual else ""
            click.echo(
                f"  {ok} Court {pred.court} {pred.court_type}: "
                f"pred={pred.predicted_winner} ({pred.confidence}){actual_str}"
            )
    click.echo("=" * 60 + "\n")


@cli.command("update-wtn")
def update_wtn() -> None:
    """
    Fetch WTN ratings for all players via the ITF public GraphQL API.

    Searches worldtennisnumber.com by player name — no login required.
    Falls back to manual entry for any player not found.
    """
    from tennisai.modules.players.store import save_player, load_all_players
    from tennisai.tools.usta_wtn import fetch_wtn_batch

    all_pf = load_all_players()
    if not all_pf:
        click.echo("No player files found. Run 'update-players' first.")
        return

    click.echo(f"\nFetching WTN for {len(all_pf)} player(s) via ITF GraphQL API...")

    batch_results = fetch_wtn_batch([pf.name for pf in all_pf])
    for pf in all_pf:
        result = batch_results.get(pf.name, {})
        s, d = result.get("singles"), result.get("doubles")
        changed = False
        if s is not None:
            pf.wtn_singles = s
            changed = True
        if d is not None:
            pf.wtn_doubles = d
            changed = True
        if changed:
            save_player(pf)
        status = f"WTN-S={pf.wtn_singles}  WTN-D={pf.wtn_doubles}" if changed else "not found"
        click.echo(f"  {pf.name}: {status}")

    # Offer manual entry for players still missing WTN
    all_pf = load_all_players()
    missing = [pf for pf in all_pf if pf.wtn_singles is None and pf.wtn_doubles is None]

    if missing:
        click.echo(f"\n{len(missing)} player(s) not found in ITF database:")
        for pf in missing:
            click.echo(f"  - {pf.name}")

        if click.confirm("\nEnter WTN manually for remaining players?", default=True):
            for pf in missing:
                click.echo(f"\n  {pf.name}")
                if pf.wtn_singles is None:
                    raw = click.prompt("    WTN Singles (Enter to skip)", default="", show_default=False).strip()
                    if raw:
                        try:
                            pf.wtn_singles = float(raw)
                        except ValueError:
                            pass
                if pf.wtn_doubles is None:
                    raw = click.prompt("    WTN Doubles (Enter to skip)", default="", show_default=False).strip()
                    if raw:
                        try:
                            pf.wtn_doubles = float(raw)
                        except ValueError:
                            pass
                save_player(pf)
                click.echo(f"    Saved: WTN-S={pf.wtn_singles}  WTN-D={pf.wtn_doubles}")

    all_pf = load_all_players()
    have_wtn = sum(1 for p in all_pf if p.wtn_singles is not None or p.wtn_doubles is not None)
    click.echo(f"\nDone. {have_wtn}/{len(all_pf)} players have WTN data.")


@cli.command("update-players")
@click.option("--team-url", default=None, help="tennisrecord.com team URL (defaults to MY_TEAM_URL in .env)")
@click.option("--with-conclusions", is_flag=True, default=False,
              help="Generate AI conclusions for each player (uses LLM tokens)")
@click.option("--with-wtn", is_flag=True, default=False,
              help="Re-fetch WTN ratings for players that have a stored USTA profile URL")
def update_players(team_url: Optional[str], with_conclusions: bool, with_wtn: bool) -> None:
    """Rebuild player files from match history and optionally update AI conclusions."""
    from tennisai.modules.players.store import rebuild_from_matches
    from tennisai.modules.players.analyzer import refresh_player_conclusions
    from tennisai.tools.tennisrecord import get_team_ratings

    team_url = _resolve_team_url(team_url)

    click.echo("\nRebuilding player files from match history...")
    players = rebuild_from_matches()
    click.echo(f"  Updated {len(players)} player file(s).")

    # Enrich with latest tennisrecord.com ratings — create files for new players too
    try:
        click.echo("Fetching latest ratings from tennisrecord.com...")
        from tennisai.modules.players.models import PlayerFile
        from tennisai.modules.players.store import load_player, save_player
        team = get_team_ratings(team_url)
        created = 0
        for p in team.players:
            pf = load_player(p.name)
            if not pf:
                pf = PlayerFile(name=p.name)
                created += 1
            pf.tennisrecord_rating = p.rating or pf.tennisrecord_rating
            pf.ntrp_level = p.ntrp_level or pf.ntrp_level
            pf.profile_url = p.profile_url or pf.profile_url
            pf.team = team.name
            save_player(pf)
        action = f"created {created} new, updated {len(team.players) - created} existing"
        click.echo(f"  {action} player file(s).")
    except Exception as exc:
        click.echo(f"  Could not fetch ratings: {exc}")

    if with_wtn:
        from tennisai.modules.players.store import load_all_players, save_player as _save_player
        from tennisai.tools.usta_wtn import fetch_wtn_batch
        all_wtn_players = load_all_players()
        if all_wtn_players:
            click.echo(f"\nRefreshing WTN for {len(all_wtn_players)} player(s) via USTA player search...")
            try:
                batch = fetch_wtn_batch([p.name for p in all_wtn_players])
                for pf in all_wtn_players:
                    result = batch.get(pf.name, {})
                    s, d = result.get("singles"), result.get("doubles")
                    changed = False
                    if s is not None:
                        pf.wtn_singles = s
                        changed = True
                    if d is not None:
                        pf.wtn_doubles = d
                        changed = True
                    if changed:
                        _save_player(pf)
                    status = f"WTN-S={pf.wtn_singles} WTN-D={pf.wtn_doubles}" if changed else "not found"
                    click.echo(f"  {pf.name}: {status}")
            except Exception as exc:
                click.echo(f"  WTN fetch failed: {exc}")
        else:
            click.echo("\nNo player files found. Run 'update-players' first.")

    if with_conclusions:
        from tennisai.modules.players.store import load_all_players
        all_players = load_all_players()
        click.echo(f"\nGenerating AI conclusions for {len(all_players)} player(s)...")
        for pf in all_players:
            click.echo(f"  {pf.name}...", nl=False)
            refresh_player_conclusions(pf.name)
            click.echo(" done")
    else:
        click.echo("\nTip: Run with --with-conclusions to generate AI player summaries.")
        click.echo("Tip: Run with --with-wtn to refresh WTN ratings from stored USTA profile URLs.")

    click.echo("\nPlayer files updated. View them in the players/ directory.")


@cli.command("backfill")
@click.option("--team-url", default=None, help="tennisrecord.com team URL (defaults to MY_TEAM_URL in .env)")
@click.option("--usta-url", default=None, help="USTA TennisLink URL (defaults to USTA_TEAM_URL in .env)")
@click.option("--all", "run_all", is_flag=True, default=False, help="Process all available past matches automatically")
def backfill(team_url: Optional[str], usta_url: Optional[str], run_all: bool) -> None:
    """Run predictions on historic matches to calibrate accuracy. Fetches lineups from USTA scorecards."""
    try:
        team_url = _resolve_team_url(team_url)
        usta_url = _resolve_usta_url(usta_url)

        from tennisai.config import get_singles_courts_override
        from tennisai.modules.matches.store import save_match, find_by_date_opponent, create_stub
        from tennisai.modules.matches.models import MatchFile
        from tennisai.modules.results import predict_match_results, apply_learnings
        from tennisai.tools.history import record_prediction
        from tennisai.tools.usta import USTAClient

        singles = get_singles_courts_override() or 2
        doubles = 3
        court_slots = _build_court_slots(singles, doubles)

        from tennisai.tools.tennisrecord import get_team_ratings, get_league_teams

        # Fetch our roster and all league teams upfront (needed for lineup prompts)
        roster: list[str] = []
        our_player_names: set[str] = set()
        league_teams = []
        try:
            click.echo("\nFetching team roster and league data from tennisrecord.com...")
            our_team_obj = get_team_ratings(team_url)
            roster = [p.name for p in our_team_obj.players]
            our_player_names = {p.name for p in our_team_obj.players}
            league_teams = get_league_teams(team_url)
            click.echo(f"  {len(roster)} players, {len(league_teams)} league teams found.")
        except Exception as exc:
            click.echo(f"  Could not fetch roster: {exc}")

        click.echo("Fetching past match scorecards from USTA TennisLink...")
        client = USTAClient()
        scorecard_entries = client.get_scorecard_urls(usta_url)

        if not scorecard_entries:
            click.echo("No past match scorecards found on USTA TennisLink.", err=True)
            client.close()
            return

        # Create offline stubs for any new matches and check existing ones
        stub_map: dict[tuple, MatchFile] = {}
        for d, opp, url in scorecard_entries:
            existing = find_by_date_opponent(d, opp)
            if existing:
                stub_map[(d, opp)] = existing
            else:
                stub = create_stub(match_date=d, opponent=opp, scorecard_url=url)
                stub_map[(d, opp)] = stub

        # Determine which matches to process
        if run_all:
            to_process = scorecard_entries
        else:
            click.echo(f"\n--- Past matches with scorecards ({len(scorecard_entries)}) ---")
            for i, (d, opp, _url) in enumerate(scorecard_entries, 1):
                stub = stub_map.get((d, opp))
                status_tag = ""
                if stub and stub.status == "completed":
                    status_tag = "  [done]"
                elif stub and stub.actual_results:
                    status_tag = "  [results recorded]"
                click.echo(f"  {i:>2}. {d}  vs {opp}{status_tag}")
            click.echo(f"  {len(scorecard_entries) + 1:>2}. Process ALL (skip already completed)")

            while True:
                raw = click.prompt("\nSelect match number").strip()
                if raw.isdigit():
                    n = int(raw)
                    if n == len(scorecard_entries) + 1:
                        to_process = scorecard_entries
                        break
                    elif 1 <= n <= len(scorecard_entries):
                        to_process = [scorecard_entries[n - 1]]
                        break
                click.echo(f"  Enter 1–{len(scorecard_entries) + 1}.")

        total_correct = 0
        total_courts = 0

        for match_date, opponent, scorecard_url in to_process:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Processing: {match_date}  vs {opponent}")
            click.echo("=" * 60)

            stub = stub_map.get((match_date, opponent))
            if stub and stub.status == "completed" and stub.actual_results:
                click.echo("  Already backfilled — skipping. (Use view-match to review.)")
                continue

            # Look up opponent roster from tennisrecord.com
            opp_roster: list[str] = []
            try:
                opp_team_obj = next(
                    (t for t in league_teams
                     if opponent.lower()[:8] in t.name.lower() and t.url != team_url),
                    None,
                )
                if opp_team_obj:
                    opp_players = get_team_ratings(opp_team_obj.url).players
                    opp_roster = [p.name for p in opp_players if p.name not in our_player_names]
                    click.echo(f"  Opponent roster: {len(opp_roster)} players from tennisrecord.com")
            except Exception:
                pass

            # Try to scrape scorecard
            click.echo("  Fetching scorecard...")
            scorecard = client.get_match_scorecard(usta_url, scorecard_url)

            our_lineup = scorecard.get("our_lineup", {})
            opp_lineup = scorecard.get("opponent_lineup", {})
            actual_from_scorecard: list = scorecard.get("actual_results", [])

            # If scraping incomplete, fall back to manual entry
            if not our_lineup:
                click.echo("  Could not parse our lineup from scorecard — entering manually.")
                our_lineup = _collect_lineup(roster, court_slots)

            if not opp_lineup:
                click.echo("  Opponent lineup not found in scorecard.")
                if click.confirm("  Enter opponent lineup manually?", default=True):
                    opp_lineup = _collect_lineup(opp_roster, court_slots)

            if our_lineup:
                click.echo("\n  Our lineup (from scorecard):")
                for court, players in our_lineup.items():
                    click.echo(f"    {court}: {', '.join(players)}")
            if opp_lineup:
                click.echo("  Opponent lineup (from scorecard):")
                for court, players in opp_lineup.items():
                    click.echo(f"    {court}: {', '.join(players)}")

            if not our_lineup:
                click.echo("  Skipping — no lineup data available.")
                continue

            # Run prediction (without revealing actual result)
            click.echo("\n  Running prediction...")
            history_text = format_history_for_prompt(get_recent_records(3))
            try:
                analysis = predict_match_results(
                    my_team_url=team_url,
                    usta_team_url=usta_url,
                    lineup=our_lineup,
                    opponent_lineup=opp_lineup or None,
                    history_text=history_text,
                    singles_courts=singles,
                    doubles_courts=doubles,
                    match_date=match_date,
                    opponent_name=opponent,
                )
            except Exception as exc:
                click.echo(f"  Prediction failed: {exc}")
                continue

            # Save prediction into the stub (or create a new record if no stub)
            if stub:
                from tennisai.modules.matches.store import save_match as _save_match
                us_wins = sum(1 for p in analysis.predictions if p.predicted_winner.lower() == "us")
                them_wins = len(analysis.predictions) - us_wins
                overall = "win" if us_wins > them_wins else ("loss" if them_wins > us_wins else "unknown")
                stub.our_lineup = our_lineup
                stub.opponent_lineup = opp_lineup or {}
                stub.predictions = analysis.predictions
                stub.overall_predicted = overall
                stub.status = "completed"
                _save_match(stub)
                match_id = stub.id
            else:
                match_record = record_prediction(analysis, our_lineup, opp_lineup or {})
                match_id = match_record.id

            # Show prediction briefly
            click.echo(f"\n  Predicted: {analysis.overall_outlook or 'see courts below'}")
            for pred in analysis.predictions:
                click.echo(f"    Court {pred.court} {pred.court_type}: {pred.predicted_winner} ({pred.confidence})")

            # Determine actual results
            actual_results: list = []
            if actual_from_scorecard:
                click.echo("\n  Actual results (from scorecard):")
                for r in actual_from_scorecard:
                    click.echo(f"    Court {r.court} {r.court_type}: {r.winner}")
                actual_results = actual_from_scorecard
            else:
                # Manual entry
                click.echo("\n  Enter actual results for each court.")
                for pred in analysis.predictions:
                    while True:
                        raw = click.prompt(
                            f"    Court {pred.court} {pred.court_type} winner [us/them/skip]",
                            default="skip", show_default=False
                        ).strip().lower()
                        if raw in ("skip", ""):
                            break
                        if raw in ("us", "them"):
                            from tennisai.models import CourtResult
                            actual_results.append(CourtResult(
                                court=pred.court,
                                court_type=pred.court_type,
                                winner=raw,
                                score="",
                            ))
                            break
                        click.echo("  Enter 'us', 'them', or 'skip'.")

            if not actual_results:
                click.echo("  No results recorded — skipping calibration.")
                continue

            # Save actual results and apply learnings
            updated = update_result(match_id, actual_results)
            if updated:
                match_file = load_match(match_id)
                if match_file:
                    apply_learnings(match_file)

                # Tally accuracy for this match
                for actual in updated.actual_results:
                    pred = next(
                        (p for p in updated.predictions
                         if p.court == actual.court and p.court_type == actual.court_type),
                        None,
                    )
                    if pred:
                        total_courts += 1
                        if pred.predicted_winner.lower() == actual.winner.lower():
                            total_correct += 1

                correct = sum(
                    1 for a in updated.actual_results
                    for p in updated.predictions
                    if p.court == a.court and p.court_type == a.court_type
                    and p.predicted_winner.lower() == a.winner.lower()
                )
                total_match = len(updated.actual_results)
                click.echo(f"\n  Accuracy this match: {correct}/{total_match} courts correct.")

        client.close()

        if total_courts > 0:
            pct = round(100 * total_correct / total_courts)
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Backfill complete. Overall accuracy: {total_correct}/{total_courts} courts ({pct}%).")
            click.echo("Player calibration files updated — future predictions will use this data.")
        else:
            click.echo("\nBackfill complete. No results recorded.")

    except EnvironmentError as exc:
        click.echo(f"\nConfiguration error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@cli.command("accuracy")
def accuracy() -> None:
    """Show prediction accuracy stats from recorded match history."""
    matches = load_all_matches()
    completed = [m for m in matches if m.actual_results and m.predictions]
    if not completed:
        click.echo("No completed matches with predictions found. Run 'record-result' or 'backfill' first.")
        return

    overall_correct = 0
    overall_total = 0
    by_court_type: dict[str, dict] = {}  # "Singles" | "Doubles" → {correct, total}
    by_court_num: dict[tuple[int, str], dict] = {}  # (num, type) → {correct, total}
    player_stats: dict[str, dict] = {}  # name → {correct, total}

    for m in completed:
        for actual in m.actual_results:
            pred = next(
                (p for p in m.predictions
                 if p.court == actual.court and p.court_type == actual.court_type),
                None,
            )
            if not pred:
                continue
            correct = pred.predicted_winner.lower() == actual.winner.lower()
            overall_total += 1
            if correct:
                overall_correct += 1

            # By court type
            ct = actual.court_type
            by_court_type.setdefault(ct, {"correct": 0, "total": 0})
            by_court_type[ct]["total"] += 1
            if correct:
                by_court_type[ct]["correct"] += 1

            # By court number + type
            key = (actual.court, actual.court_type)
            by_court_num.setdefault(key, {"correct": 0, "total": 0})
            by_court_num[key]["total"] += 1
            if correct:
                by_court_num[key]["correct"] += 1

            # By player
            for name in pred.my_players:
                player_stats.setdefault(name, {"correct": 0, "total": 0})
                player_stats[name]["total"] += 1
                if correct:
                    player_stats[name]["correct"] += 1

    click.echo(f"\n{'=' * 60}")
    click.echo("PREDICTION ACCURACY")
    click.echo("=" * 60)
    click.echo(f"Matches analyzed: {len(completed)}")
    if overall_total:
        pct = round(100 * overall_correct / overall_total)
        click.echo(f"Overall courts:   {overall_correct}/{overall_total} ({pct}%)")
    else:
        click.echo("Overall courts:   no data")

    click.echo("\n--- By Court Type ---")
    for ct, stats in sorted(by_court_type.items()):
        pct = round(100 * stats["correct"] / stats["total"]) if stats["total"] else 0
        click.echo(f"  {ct}: {stats['correct']}/{stats['total']} ({pct}%)")

    click.echo("\n--- By Court Number ---")
    for (num, ct), stats in sorted(by_court_num.items()):
        pct = round(100 * stats["correct"] / stats["total"]) if stats["total"] else 0
        click.echo(f"  Court {num} {ct}: {stats['correct']}/{stats['total']} ({pct}%)")

    if player_stats:
        click.echo("\n--- By Player (courts predicted) ---")
        for name, stats in sorted(player_stats.items(), key=lambda x: -x[1]["total"]):
            pct = round(100 * stats["correct"] / stats["total"]) if stats["total"] else 0
            click.echo(f"  {name:<30} {stats['correct']}/{stats['total']} ({pct}%)")

    click.echo("=" * 60 + "\n")


@cli.command("list-teams")
def list_teams() -> None:
    """List all configured teams and show which one is active."""
    teams = get_all_teams()
    active = get_active_team_index()

    if not teams:
        click.echo("No teams configured. Set TEAM_COUNT and TEAM_N_NAME/URL/USTA_URL in .env.")
        return

    click.echo("\n--- Configured teams ---")
    for t in teams:
        marker = "*" if t["index"] == active else " "
        name = t["name"] or f"Team {t['index']}"
        click.echo(f"  [{marker}] {t['index']}. {name}")
    click.echo()
    click.echo("Use 'switch-team <number>' to change the active team.")


@cli.command("switch-team")
@click.argument("team_num", required=False, type=int)
def switch_team(team_num: Optional[int]) -> None:
    """Switch the active team (persists to .env)."""
    teams = get_all_teams()
    active = get_active_team_index()

    if not teams:
        click.echo("No teams configured.", err=True)
        sys.exit(1)

    click.echo("\n--- Configured teams ---")
    for t in teams:
        marker = "*" if t["index"] == active else " "
        name = t["name"] or f"Team {t['index']}"
        click.echo(f"  [{marker}] {t['index']}. {name}")
    click.echo()

    if team_num is None:
        raw = click.prompt("Switch to team number").strip()
        if not raw.isdigit():
            click.echo("Invalid team number.", err=True)
            sys.exit(1)
        team_num = int(raw)

    valid = [t["index"] for t in teams]
    if team_num not in valid:
        click.echo(f"Team {team_num} not found. Valid options: {valid}", err=True)
        sys.exit(1)

    set_active_team(team_num)
    new_cfg = next(t for t in teams if t["index"] == team_num)
    click.echo(f"Switched to team {team_num}: {new_cfg['name'] or f'Team {team_num}'}")
    click.echo("Run 'analyze' or 'suggest-lineup' to work with this team.")


def main() -> None:
    cli()
