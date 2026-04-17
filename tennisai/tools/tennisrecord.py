"""Scraper for tennisrecord.com team and league pages."""

import datetime
import re
from typing import Optional
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin

from tennisai.models import Player, PlayerHistory, Team

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TennisAI/1.0)"
}
_BASE = "https://www.tennisrecord.com"


def _get(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _col_index(headers: list[str], *candidates: str) -> Optional[int]:
    """Return the index of the first matching header name (case-insensitive)."""
    lower = [h.lower().strip() for h in headers]
    for c in candidates:
        try:
            return lower.index(c.lower())
        except ValueError:
            continue
    return None


def get_team_ratings(team_url: str) -> Team:
    """
    Scrape player names, NTRP level, and estimated rating from a tennisrecord.com team profile.

    tennisrecord.com shows two values per player:
      - NTRP: the official band (e.g. 3.0, 3.5)
      - Rating: the estimated current rating within that band, shown to 2 decimal places (e.g. 2.98)
                This is the higher-precision predictor and should be preferred when comparing players.
    """
    soup = _get(team_url)

    title_tag = soup.find("h1") or soup.find("h2")
    team_name = title_tag.get_text(strip=True) if title_tag else "Unknown Team"

    players: list[Player] = []

    # Find the player table by looking for one that has both "Name" and "Rating" headers
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers:
            continue

        name_idx = _col_index(headers, "name")
        ntrp_idx = _col_index(headers, "ntrp")
        rating_idx = _col_index(headers, "rating")

        if name_idx is None or rating_idx is None:
            continue

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) <= rating_idx:
                continue

            name = cells[name_idx].get_text(strip=True)
            if not name:
                continue

            ntrp_raw = cells[ntrp_idx].get_text(strip=True) if ntrp_idx is not None else ""
            rating_raw = cells[rating_idx].get_text(strip=True)

            ntrp_val: Optional[float] = None
            if re.match(r"^\d+\.\d+$", ntrp_raw):
                ntrp_val = float(ntrp_raw)

            rating_val: Optional[float] = None
            if re.match(r"^\d+\.\d+$", rating_raw):
                rating_val = float(rating_raw)

            # Only include rows where we got at least one numeric value
            if ntrp_val is not None or rating_val is not None:
                # Extract player profile link if the name cell has an <a> tag
                profile_url = ""
                name_cell = cells[name_idx]
                link = name_cell.find("a", href=True)
                if link:
                    href = link["href"]
                    profile_url = href if href.startswith("http") else urljoin(_BASE, href)

                players.append(Player(
                    name=name,
                    ntrp_level=ntrp_val,
                    rating=rating_val,
                    team=team_name,
                    profile_url=profile_url,
                ))

        if players:
            break  # stop after the first matching table

    return Team(name=team_name, url=team_url, players=players)


def get_league_teams(team_url: str) -> list[Team]:
    """
    From a team profile page, find the league page URL and return all teams in that league
    (Team objects with URL; players not yet fetched).
    """
    soup = _get(team_url)

    league_url: Optional[str] = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "league.aspx" in href or "flight" in href.lower():
            league_url = href if href.startswith("http") else urljoin(_BASE, href)
            break

    if not league_url:
        raise ValueError(
            f"Could not find a league/flight link on the team page: {team_url}\n"
            "The page structure may have changed. Please check tennisrecord.com manually."
        )

    return _scrape_league_page(league_url)


def get_player_history(player_url: str, months: int = 6) -> PlayerHistory:
    """
    Scrape a player's match history from their tennisrecord.com profile page.
    Returns wins/losses and individual match records within the last `months` months.
    """
    soup = _get(player_url)

    name_tag = soup.find("h1") or soup.find("h2")
    player_name = name_tag.get_text(strip=True) if name_tag else "Unknown Player"

    cutoff = datetime.date.today() - datetime.timedelta(days=months * 30)
    matches: list[dict] = []
    wins = 0
    losses = 0

    # tennisrecord.com player pages have a match history table
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        # Look for a table with date and result/score columns
        has_date = any("date" in h for h in headers)
        has_result = any(h in ("w/l", "result", "score", "won") for h in headers)
        if not has_date and not has_result:
            continue

        date_idx = next((i for i, h in enumerate(headers) if "date" in h), None)
        result_idx = next((i for i, h in enumerate(headers) if h in ("w/l", "result", "won")), None)
        score_idx = next((i for i, h in enumerate(headers) if "score" in h), None)
        opp_idx = next((i for i, h in enumerate(headers) if "opponent" in h or "team" in h), None)

        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells:
                continue

            match_date = None
            if date_idx is not None and date_idx < len(cells):
                for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
                    try:
                        match_date = datetime.datetime.strptime(cells[date_idx], fmt).date()
                        break
                    except ValueError:
                        continue

            if match_date and match_date < cutoff:
                continue

            result_text = cells[result_idx].upper() if result_idx is not None and result_idx < len(cells) else ""
            won: Optional[bool] = None
            if result_text.startswith("W"):
                won = True
                wins += 1
            elif result_text.startswith("L"):
                won = False
                losses += 1

            score = cells[score_idx] if score_idx is not None and score_idx < len(cells) else ""
            opponent = cells[opp_idx] if opp_idx is not None and opp_idx < len(cells) else ""

            matches.append({
                "date": str(match_date) if match_date else "",
                "opponent": opponent,
                "won": won,
                "score": score,
            })

        if matches:
            break

    return PlayerHistory(
        player_name=player_name,
        wins_last_6_months=wins,
        losses_last_6_months=losses,
        matches=matches,
    )


def _scrape_league_page(league_url: str) -> list[Team]:
    """Scrape all team links from a tennisrecord.com league/flight page."""
    soup = _get(league_url)
    teams: list[Team] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "teamprofile.aspx" in href:
            full_url = href if href.startswith("http") else urljoin(_BASE, href)
            team_name = a.get_text(strip=True)
            if team_name:
                teams.append(Team(name=team_name, url=full_url))

    return teams
