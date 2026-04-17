"""
Tennis AI CLI — entry point.

Usage (URLs configured in .env):
    python -m tennisai analyze
    python -m tennisai analyze --output-csv results.csv

Usage (URLs as arguments):
    python -m tennisai analyze --team-url <url> --usta-url <url>
"""

import csv
import sys
from typing import Optional

import click

from tennisai.agent import run_analysis
from tennisai.config import get_my_team_url, get_usta_team_url
from tennisai.models import MatchAnalysis


def _resolve_team_url(option_value: Optional[str]) -> str:
    if option_value:
        return option_value
    return get_my_team_url()


def _resolve_usta_url(option_value: Optional[str]) -> str:
    if option_value:
        return option_value
    return get_usta_team_url()

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
            winner = "US" if pred.predicted_winner.lower() == "us" else "THEM"
            click.echo(
                f"\nCourt {pred.court} {pred.court_type}"
                f"\n  Us:     {players_us}"
                f"\n  Them:   {players_them}"
                f"\n  Pick:   {winner} ({pred.confidence} confidence)"
                f"\n  Why:    {pred.reasoning}"
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
            "predicted_winner", "confidence", "reasoning",
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
def analyze(team_url: Optional[str], usta_url: Optional[str], output_csv: Optional[str]) -> None:
    """Analyze your next USTA match and predict court outcomes."""
    try:
        team_url = _resolve_team_url(team_url)
        usta_url = _resolve_usta_url(usta_url)

        from tennisai.tools.tennisrecord import get_team_ratings
        from tennisai.tools.usta import USTAClient

        click.echo("Fetching team roster from tennisrecord.com...")
        try:
            team = get_team_ratings(team_url)
            roster = [p.name for p in team.players]
        except Exception:
            roster = []

        if not roster:
            click.echo("Could not fetch roster — you can still type player names directly.\n")

        click.echo("Fetching match format from USTA TennisLink...")
        try:
            usta_client = USTAClient()
            match_format = usta_client.get_match_format(usta_url)
            usta_client.close()
            singles = match_format.singles_courts
            doubles = match_format.doubles_courts
            click.echo(f"  Match format: {singles} Singles, {doubles} Doubles")
        except Exception:
            singles, doubles = 3, 3
            click.echo("  Could not detect format — assuming 3 Singles, 3 Doubles.")

        court_slots = _build_court_slots(singles, doubles)
        lineup = _collect_lineup(roster, court_slots)
        if not lineup:
            click.echo("No lineup entered. At least one court must have a player assigned.", err=True)
            sys.exit(1)

        click.echo("\nAnalyzing match... (this may take a moment)")
        analysis = run_analysis(
            my_team_url=team_url,
            usta_team_url=usta_url,
            lineup=lineup,
        )

        _print_analysis(analysis)

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


def main() -> None:
    cli()
