"""
USTA TennisLink client using Playwright.

TennisLink uses Auth0 for authentication and JavaScript to render content,
so we use a headless Chromium browser to log in and scrape the data.

We intercept all JSON network responses the page makes after login — this lets
us capture the actual API data without needing to know endpoint URLs in advance.

Setup (one-time):
    pip install playwright
    playwright install chromium
"""

import json
import re
import datetime
from typing import Optional

from tennisai.config import get_my_team_name, get_usta_credentials
from tennisai.models import CourtTrend, Match, MatchFormat, MatchResult

# How long (ms) to wait for the login redirect back to tennislink after submitting credentials
LOGIN_TIMEOUT_MS = 30_000
# How long to wait for the page to finish making network requests after login
NETWORK_IDLE_TIMEOUT_MS = 15_000


def _launch_and_login(usta_url: str):
    """
    Launch a headless Chromium browser, log in via Auth0, navigate to usta_url,
    and return (page, browser, captured_responses).

    captured_responses is a list of {"url": str, "data": any} for every JSON
    response the page received — this is how we find the schedule/results API data.
    """
    from playwright.sync_api import sync_playwright

    username, password = get_usta_credentials()
    captured: list[dict] = []

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    def _on_response(response):
        ct = response.headers.get("content-type", "")
        if "json" in ct:
            try:
                captured.append({"url": response.url, "data": response.json()})
            except Exception:
                pass

    page.on("response", _on_response)

    # Navigate — this will redirect to Auth0 login
    page.goto(usta_url, wait_until="domcontentloaded")

    # Auth0 login form — press Enter rather than clicking the submit button,
    # because Auth0 ULP overlays a hidden button that intercepts pointer events.
    page.wait_for_selector('input[name="username"]', timeout=LOGIN_TIMEOUT_MS)
    page.fill('input[name="username"]', username)
    page.fill('input[name="password"]', password)
    page.press('input[name="password"]', 'Enter')

    # Wait for the entire Auth0 redirect chain to finish (can be 3-4 hops).
    # networkidle fires once all redirects settle and no network requests are in-flight.
    page.wait_for_load_state("networkidle", timeout=LOGIN_TIMEOUT_MS)

    if "tennislink.usta.com" not in page.url:
        page.screenshot(path="/tmp/usta_login_debug.png")
        raise PermissionError(
            f"USTA login did not land on tennislink.usta.com (ended up at {page.url}).\n"
            "Screenshot saved to /tmp/usta_login_debug.png — check for wrong password, "
            "CAPTCHA, or MFA prompt."
        )

    # Dismiss cookie consent banner if present (clicks the first Accept/OK button)
    for selector in ('button:has-text("Accept")', 'button:has-text("OK")',
                     'button:has-text("Accept All")', '[id*="cookie"] button',
                     '[class*="cookie"] button'):
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=2_000):
                btn.click()
                break
        except Exception:
            pass

    # Wait for the JS app to finish loading schedule/standings data
    try:
        page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
    except Exception:
        pass  # networkidle can time out on slow pages; captured data is still usable

    return page, browser, pw, captured


def _close(page, browser, pw) -> None:
    try:
        browser.close()
        pw.stop()
    except Exception:
        pass


class USTAClient:
    def __init__(self) -> None:
        self._logged_in = False
        # Playwright objects kept for potential reuse within a single session
        self._page = None
        self._browser = None
        self._pw = None
        self._captured: list[dict] = []

    def login(self) -> None:
        """Verify USTA credentials are configured (actual Playwright login happens on first use)."""
        from tennisai.config import get_usta_credentials
        get_usta_credentials()  # raises EnvironmentError if missing
        # Actual browser login happens lazily in _ensure_page()

    def _ensure_page(self, usta_url: str) -> None:
        """Open browser, log in, then navigate to the team-specific league page."""
        if self._page is not None:
            return

        self._page, self._browser, self._pw, self._captured = _launch_and_login(usta_url)
        self._logged_in = True

        # After Auth0 redirects back, the hash fragment (s=TOKEN) is lost.
        # Navigate back to the original URL so the SPA JavaScript processes the
        # correct team token and loads that team's schedule/results data.
        self._page.goto(usta_url, wait_until="domcontentloaded")
        try:
            self._page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
        except Exception:
            pass

    def close(self) -> None:
        if self._page:
            _close(self._page, self._browser, self._pw)
            self._page = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_schedule_and_results(self, usta_url: str) -> tuple[list[Match], list[MatchResult]]:
        """
        Return (upcoming_matches, recent_results) for the team.
        Each tab replaces the page content, so we capture HTML after each click separately.
        """
        self._ensure_page(usta_url)

        # Click Match Schedule tab and capture HTML for upcoming matches
        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchScheduleForTeams")
        schedule_html = self._page.content()

        # Click Match Summary tab and capture HTML for completed results
        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchSummaryForTeams")
        summary_html = self._page.content()

        upcoming, _ = _parse_from_html(schedule_html)
        _, results = _parse_from_html(summary_html)

        # Fallback: if summary tab returned nothing, extract completed rows from schedule tab
        if not results:
            _, results = _parse_completed_from_schedule(schedule_html)

        return upcoming, results

    def get_schedule(self, usta_url: str) -> list[Match]:
        upcoming, _ = self.get_schedule_and_results(usta_url)
        return upcoming

    def get_next_match(self, usta_url: str) -> Optional[Match]:
        matches = self.get_schedule(usta_url)
        today = datetime.date.today()
        upcoming = [m for m in matches if m.date and m.date >= today]
        return upcoming[0] if upcoming else None

    def get_recent_results(self, usta_url: str) -> list[MatchResult]:
        _, results = self.get_schedule_and_results(usta_url)
        return results

    def get_match_format(self, usta_url: str) -> MatchFormat:
        """
        Determine the match format by reading a completed match scorecard.
        Scorecards list every court played, so counting them is definitive.
        Falls back to scanning page text if no scorecard links are found.
        """
        self._ensure_page(usta_url)

        # Get scorecard links from Match Summary tab
        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchSummaryForTeams")
        summary_html = self._page.content()
        scorecard_urls = _extract_scorecard_urls(summary_html)

        for url in scorecard_urls[:3]:
            try:
                full_url = url if url.startswith("http") else f"https://tennislink.usta.com{url}"
                self._page.goto(full_url, wait_until="domcontentloaded")
                try:
                    self._page.wait_for_load_state("networkidle", timeout=8_000)
                except Exception:
                    pass
                fmt = _detect_format_from_scorecard(self._page.content())
                if fmt:
                    # Navigate back so subsequent calls still work
                    self._page.goto(usta_url, wait_until="domcontentloaded")
                    try:
                        self._page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
                    except Exception:
                        pass
                    return fmt
            except Exception:
                continue

        # Fallback: scan the schedule tab for explicit format text
        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchScheduleForTeams")
        return _parse_match_format(self._page.content())

    def get_court_trends(self, usta_url: str) -> list[CourtTrend]:
        """
        Return per-court win/loss trends by scraping individual match scorecards.
        Visits each 'View Score' link from the Match Summary tab.
        """
        self._ensure_page(usta_url)

        # Get Match Summary HTML for scorecard links
        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchSummaryForTeams")
        summary_html = self._page.content()

        scorecard_urls = _extract_scorecard_urls(summary_html)
        court_stats: dict[tuple[int, str], list[bool]] = {}  # (court_num, type) → [won, ...]

        for url in scorecard_urls[:8]:  # cap at 8 most recent matches
            try:
                full_url = url if url.startswith("http") else f"https://tennislink.usta.com{url}"
                self._page.goto(full_url, wait_until="domcontentloaded")
                try:
                    self._page.wait_for_load_state("networkidle", timeout=8_000)
                except Exception:
                    pass
                card_html = self._page.content()
                court_results = _parse_scorecard(card_html)
                for (court_num, court_type), won in court_results:
                    key = (court_num, court_type)
                    court_stats.setdefault(key, []).append(won)
            except Exception:
                continue

        # Navigate back to usta_url so subsequent tab clicks still work
        try:
            self._page.goto(usta_url, wait_until="domcontentloaded")
            self._page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
        except Exception:
            pass

        trends: list[CourtTrend] = []
        for (court_num, court_type), results in sorted(court_stats.items()):
            trends.append(CourtTrend(
                court=court_num,
                court_type=court_type,
                wins=sum(1 for w in results if w),
                losses=sum(1 for w in results if not w),
            ))
        return trends

    def get_standings(self, usta_url: str) -> dict:
        """
        Fetch current league standings from the USTA Standings tab.
        Returns list of teams with their position, wins, losses, and total matches.
        Also returns our team's position and matches remaining in the season.
        """
        self._ensure_page(usta_url)
        _click_single_tab(self._page, "ctl00_mainContent_lnkStandingsForTeams")
        html = self._page.content()
        return _parse_standings(html)

    def get_roster_with_profiles(self, usta_url: str) -> dict[str, dict]:
        """
        Scrape our team's roster tab.
        Returns dict keyed by player name:
          {"profile_url": str, "wtn_singles": float|None, "wtn_doubles": float|None, "ntrp": float|None}
        WTN is read directly from the roster table when available, saving individual profile fetches.
        Profile URL is stored so fetch_wtn() can be called later if WTN is not in the table.
        """
        self._ensure_page(usta_url)
        _click_single_tab(self._page, "ctl00_mainContent_lnkRosterForTeams")
        html = self._page.content()
        return _parse_roster_html(html)

    def get_opponent_profiles_from_scorecards(self, usta_url: str) -> dict[str, dict]:
        """
        Visit recent match scorecards and extract opponent player profile links.
        Returns same structure as get_roster_with_profiles.
        Uses at most 5 recent scorecards to cap Playwright time.
        """
        self._ensure_page(usta_url)

        # Collect scorecard URLs from both tabs
        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchSummaryForTeams")
        scorecard_urls = _extract_scorecard_urls(self._page.content())
        if not scorecard_urls:
            _click_single_tab(self._page, "ctl00_mainContent_lnkMatchScheduleForTeams")
            scorecard_urls = _extract_scorecard_urls(self._page.content())

        players: dict[str, dict] = {}
        our_team = get_my_team_name().lower()

        for url in scorecard_urls[:5]:
            full_url = url if url.startswith("http") else f"https://tennislink.usta.com{url}"
            try:
                self._page.goto(full_url, wait_until="domcontentloaded")
                try:
                    self._page.wait_for_load_state("networkidle", timeout=8_000)
                except Exception:
                    pass
                found = _parse_scorecard_player_profiles(self._page.content(), our_team)
                for name, info in found.items():
                    if name not in players:
                        players[name] = info
            except Exception:
                continue

        # Navigate back
        try:
            self._page.goto(usta_url, wait_until="domcontentloaded")
            self._page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
        except Exception:
            pass

        return players

    def get_full_schedule(self, usta_url: str) -> tuple[list[Match], list[MatchResult]]:
        """Return both upcoming and completed matches (same as get_schedule_and_results)."""
        return self.get_schedule_and_results(usta_url)

    def get_scorecard_urls(self, usta_url: str) -> list[tuple[datetime.date, str, str]]:
        """
        Return list of (date, opponent, full_url) for all past matches with View Score links.
        Always checks both the Match Summary and Match Schedule tabs and merges results,
        because recent completed matches may appear on the Schedule tab before appearing
        on the Summary tab.
        """
        self._ensure_page(usta_url)

        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchSummaryForTeams")
        summary_entries = _extract_scorecard_urls_with_meta(self._page.content())

        _click_single_tab(self._page, "ctl00_mainContent_lnkMatchScheduleForTeams")
        sched_entries = _extract_scorecard_urls_with_meta(self._page.content())

        # Deduplicate by (date, opponent prefix) — same match can appear on both tabs
        seen_key: set[tuple] = set()
        merged: list[tuple[datetime.date, str, str]] = []
        for entry in summary_entries + sched_entries:
            key = (entry[0], entry[1].lower()[:12])
            if key not in seen_key:
                seen_key.add(key)
                merged.append(entry)

        merged.sort(key=lambda e: e[0], reverse=True)
        return merged

    def get_match_scorecard(self, usta_url: str, scorecard_url: str) -> dict:
        """
        Scrape a single USTA match scorecard.
        Returns dict with keys:
          date, home_team, away_team,
          our_lineup (dict[court_label, list[str]]),
          opponent_lineup (dict[court_label, list[str]]),
          actual_results (list[CourtResult])
        Returns empty dict on failure.
        """
        self._ensure_page(usta_url)
        full_url = scorecard_url if scorecard_url.startswith("http") else f"https://tennislink.usta.com{scorecard_url}"
        try:
            self._page.goto(full_url, wait_until="domcontentloaded")
            try:
                self._page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
            html = self._page.content()
            return _parse_scorecard_full(html)
        except Exception:
            return {}
        finally:
            # Navigate back so the session stays on the team page
            try:
                self._page.goto(usta_url, wait_until="domcontentloaded")
                self._page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
            except Exception:
                pass

    def debug_page(self, usta_url: str) -> None:
        """
        Extended debug: click every visible tab/link on the stats page to trigger
        lazy-loaded API calls, then print all captured JSON and save full HTML.
        """
        self._ensure_page(usta_url)

        print(f"\n  Current page URL: {self._page.url}")
        print("\n  Clicking visible tabs to trigger lazy-loaded data...")
        _click_all_tabs(self._page, self._captured)

        noisy = ("demdex.net", "cookielaw.org", "quantserve", "twitter.com",
                 "google.com", "adtrafficquality", "onetrust", "omtrdc",
                 "facebook.net", "account.usta.com", "Tournaments", "FlexLeagues")

        print(f"\n  All captured JSON responses ({len(self._captured)}) — filtered:")
        for i, r in enumerate(self._captured):
            if any(n in r["url"] for n in noisy):
                continue
            preview = json.dumps(r["data"])[:600]
            print(f"\n  [{i}] {r['url']}")
            print(f"       {preview}")

        # Save full HTML for inspection
        html = self._page.content()
        html_path = "/tmp/usta_page.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"\n  Full rendered HTML saved to {html_path}")


# ------------------------------------------------------------------
# Parsers — try multiple common USTA API shapes
# ------------------------------------------------------------------

def _parse_upcoming_from_captured(captured: list[dict]) -> list[Match]:
    """Look through captured JSON responses for schedule/upcoming match data."""
    matches: list[Match] = []
    today = datetime.date.today()

    for item in captured:
        data = item["data"]
        rows = _find_list(data, ("schedule", "Schedule", "matches", "Matches",
                                  "upcomingMatches", "teamSchedule"))
        for row in rows:
            d = _parse_date_flexible(
                row.get("matchDate") or row.get("MatchDate") or row.get("date") or ""
            )
            if d and d >= today:
                matches.append(Match(
                    date=d,
                    home_team=_str(row, "homeTeam", "HomeTeam", "home"),
                    away_team=_str(row, "awayTeam", "AwayTeam", "away"),
                    location=_str(row, "facility", "Facility", "location", "venue"),
                ))

    matches.sort(key=lambda m: m.date or datetime.date.min)
    return matches


def _parse_results_from_captured(captured: list[dict]) -> list[MatchResult]:
    """Look through captured JSON responses for completed match results."""
    results: list[MatchResult] = []
    today = datetime.date.today()

    for item in captured:
        data = item["data"]
        rows = _find_list(data, ("schedule", "Schedule", "matches", "Matches",
                                  "results", "Results", "teamSchedule"))
        for row in rows:
            d = _parse_date_flexible(
                row.get("matchDate") or row.get("MatchDate") or row.get("date") or ""
            )
            if not d or d >= today:
                continue  # skip upcoming or undated

            score_raw = (
                row.get("score") or row.get("Score") or
                row.get("matchScore") or row.get("result") or ""
            )
            won = _parse_win(row.get("win") or row.get("Win") or row.get("winner"), score_raw)
            opponent = (
                row.get("opponent") or row.get("Opponent") or
                row.get("awayTeam") or row.get("AwayTeam") or
                row.get("homeTeam") or row.get("HomeTeam") or ""
            )

            results.append(MatchResult(
                date=d,
                opponent=str(opponent),
                location=_str(row, "facility", "Facility", "location", "venue"),
                score=str(score_raw),
                won=won,
            ))

    results.sort(key=lambda r: r.date or datetime.date.min, reverse=True)
    return results[:5]  # most recent 5


def _parse_completed_from_schedule(html: str) -> tuple[list[Match], list[MatchResult]]:
    """
    Extract completed matches directly from the schedule tab HTML.
    Rows with 'View Score' (and not 'Enter Score') are completed matches.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results: list[MatchResult] = []
    our_team = get_my_team_name()

    sched_panel = soup.find("div", id="ctl00_mainContent_pnlMatchSchedule")
    if not sched_panel:
        # Try any table in the page as fallback
        sched_panel = soup

    for row in sched_panel.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        if len(cells) < 16:
            continue
        action = cells[8]
        if "View Score" not in action or "Enter Score" in action:
            continue
        d = _parse_date_flexible(cells[9])
        if not d:
            continue
        home = _clean_team(cells[11])
        away = _clean_team(cells[13])
        facility = _clean_team(cells[15]) if len(cells) > 15 else ""
        is_home = our_team.lower() in home.lower() if our_team else True
        opponent = away if is_home else home
        results.append(MatchResult(date=d, opponent=opponent, location=facility))

    return [], results


def _parse_from_html(html: str) -> tuple[list[Match], list[MatchResult]]:
    """
    Parse USTA TennisLink schedule and match results from rendered page HTML.

    The schedule table has rows with 16 cells. The visible data cells are:
      [9]  date        e.g. "3/27/2026"
      [10] time        e.g. "7:00 PM"
      [11] home team
      [12] home captain
      [13] visiting team
      [14] visiting captain
      [15] facility
    Cell [8] contains the action text: "View Score" means completed, "Enter Score" means upcoming.

    Match summary rows have a similar structure but include the score.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    upcoming: list[Match] = []
    results: list[MatchResult] = []
    today = datetime.date.today()

    # --- Schedule ---
    sched_panel = soup.find("div", id="ctl00_mainContent_pnlMatchSchedule")
    if sched_panel:
        for row in sched_panel.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 16:
                continue
            action = cells[8]
            d = _parse_date_flexible(cells[9])
            home = _clean_team(cells[11])
            away = _clean_team(cells[13])
            facility = _clean_team(cells[15])
            if not d or not (home or away):
                continue

            is_completed = "View Score" in action and "Enter Score" not in action
            # Include as upcoming if: action says so, OR date is in the future and not already scored
            if "Enter Score" in action or "Print Blank Score" in action or (d >= today and not is_completed):
                upcoming.append(Match(date=d, home_team=home, away_team=away, location=facility))

    # --- Match Summary (results with scores) ---
    summary_panel = soup.find("div", id=lambda x: x and "matchsummary" in x.lower()
                              and "pnl" in x.lower())
    if summary_panel:
        for row in summary_panel.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 14:
                continue
            d = _parse_date_flexible(cells[9])
            home = _clean_team(cells[11])
            away = _clean_team(cells[13])
            # Score is typically in cells[8] for summary rows, format like "3-2" or "W 3-2"
            score_raw = cells[8].strip()
            if not d:
                continue
            # Determine which side is "us" based on our team name
            our_team = get_my_team_name()
            is_home = our_team.lower() in home.lower()
            opponent = away if is_home else home
            won = _parse_win(None, score_raw)
            results.append(MatchResult(
                date=d, opponent=opponent,
                location=cells[15].strip() if len(cells) > 15 else "",
                score=score_raw, won=won,
            ))

    # If no summary panel, infer completed matches from schedule rows marked "View Score"
    if not results and sched_panel:
        for row in sched_panel.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 16:
                continue
            if "View Score" not in cells[8] or "Enter Score" in cells[8]:
                continue
            d = _parse_date_flexible(cells[9])
            home = _clean_team(cells[11])
            away = _clean_team(cells[13])
            if not d:
                continue
            our_team = get_my_team_name()
            is_home = our_team.lower() in home.lower()
            opponent = away if is_home else home
            results.append(MatchResult(date=d, opponent=opponent,
                                        location=cells[15].strip()))

    upcoming.sort(key=lambda m: m.date or datetime.date.min)
    results.sort(key=lambda r: r.date or datetime.date.min, reverse=True)
    return upcoming, results[:5]


def _detect_format_from_scorecard(html: str) -> Optional["MatchFormat"]:
    """
    Count singles and doubles courts from a rendered match scorecard page.
    Returns None if no court labels are found (page may not have loaded properly).
    """
    from bs4 import BeautifulSoup
    from tennisai.models import MatchFormat

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ")

    singles: set[int] = set()
    doubles: set[int] = set()

    for m in re.finditer(r'\bS(?:ingles)?\s*(\d)\b', text, re.IGNORECASE):
        singles.add(int(m.group(1)))
    for m in re.finditer(r'\bD(?:oubles)?\s*(\d)\b', text, re.IGNORECASE):
        doubles.add(int(m.group(1)))

    if not singles:
        return None

    return MatchFormat(
        singles_courts=max(singles),
        doubles_courts=max(doubles) if doubles else 3,
    )


def _parse_match_format(html: str) -> "MatchFormat":
    """
    Detect match format from USTA page HTML.
    Looks for patterns like "2 Singles" or "1 Singles, 3 Doubles".
    Defaults to 3 singles + 3 doubles.
    """
    from tennisai.models import MatchFormat

    text = html.lower()
    singles = 2
    doubles = 3

    # Look for explicit format text, e.g. "format: 2 singles, 3 doubles"
    m = re.search(r"(\d)\s*singles", text)
    if m:
        singles = int(m.group(1))
    m = re.search(r"(\d)\s*doubles", text)
    if m:
        doubles = int(m.group(1))

    return MatchFormat(singles_courts=singles, doubles_courts=doubles)


def _extract_scorecard_urls(html: str) -> list[str]:
    """
    Extract 'View Score' link hrefs from Match Summary HTML.
    Returns list of relative or absolute URLs to individual match scorecards.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()

    # Search every <a> whose text contains "View Score"
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if "view score" in text.lower() or "scorecard" in text.lower():
            href = a["href"]
            if href not in seen:
                seen.add(href)
                urls.append(href)

    # Also look in the summary panel by ID
    summary_panel = soup.find("div", id=lambda x: x and "matchsummary" in x.lower() and "pnl" in x.lower())
    if summary_panel:
        for a in summary_panel.find_all("a", href=True):
            href = a["href"]
            if href not in seen and ("matchid" in href.lower() or "scorecard" in href.lower()):
                seen.add(href)
                urls.append(href)

    return urls


def _extract_scorecard_urls_with_meta(html: str) -> list[tuple[datetime.date, str, str]]:
    """
    Extract (date, opponent, full_url) from USTA schedule/summary HTML.

    Mirrors _parse_completed_from_schedule exactly (cells[8] action check,
    16-cell requirement, fixed indices for date/teams) then additionally
    extracts the href for the scorecard link from the row.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    our_team = get_my_team_name()
    entries: list[tuple[datetime.date, str, str]] = []
    seen: set[str] = set()

    # Search within the schedule panel if present, otherwise whole page
    panel = (soup.find("div", id="ctl00_mainContent_pnlMatchSchedule")
             or soup.find("div", id=lambda x: x and "matchsummary" in x.lower() and "pnl" in x.lower())
             or soup)

    for row in panel.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        if len(cells) < 16:
            continue

        action = cells[8]
        if "View Score" not in action or "Enter Score" in action:
            continue

        d = _parse_date_flexible(cells[9])
        if not d:
            continue

        home = _clean_team(cells[11])
        away = _clean_team(cells[13])
        is_home = our_team.lower() in home.lower() if our_team else True
        opponent = away if is_home else home
        if not opponent:
            opponent = "Unknown"

        # Find the scorecard href — any link in the row works
        href = None
        for a in row.find_all("a", href=True):
            h = a["href"]
            # Prefer links that look like scorecards; fall back to any link
            if "score" in h.lower() or "matchid" in h.lower():
                href = h
                break
        if not href:
            for a in row.find_all("a", href=True):
                href = a["href"]
                break
        if not href:
            continue

        full_url = href if href.startswith("http") else f"https://tennislink.usta.com{href}"
        if full_url not in seen:
            seen.add(full_url)
            entries.append((d, opponent, full_url))

    entries.sort(key=lambda e: e[0], reverse=True)
    return entries


def _parse_scorecard_full(html: str) -> dict:
    """
    Parse a USTA match scorecard page into structured data.
    Returns:
      {date, home_team, away_team,
       our_lineup: dict[str, list[str]],
       opponent_lineup: dict[str, list[str]],
       actual_results: list[CourtResult]}
    """
    from bs4 import BeautifulSoup
    from tennisai.models import CourtResult

    soup = BeautifulSoup(html, "html.parser")
    our_team = get_my_team_name().lower()

    # --- Parse match header (date, teams) ---
    date_val: datetime.date | None = None
    home_team = ""
    away_team = ""
    text = soup.get_text(" ")
    for line in text.splitlines():
        line = line.strip()
        if not date_val:
            date_val = _parse_date_flexible(line)
        if not home_team and len(line) > 5 and "vs" in line.lower():
            parts = re.split(r"\s+vs\.?\s+", line, flags=re.IGNORECASE)
            if len(parts) == 2:
                home_team = parts[0].strip()
                away_team = parts[1].strip()
            break

    is_home = our_team[:6] in home_team.lower() if home_team else True

    # --- Parse per-court rows ---
    our_lineup: dict[str, list[str]] = {}
    opp_lineup: dict[str, list[str]] = {}
    actual_results: list[CourtResult] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Check if this looks like a court table (has S1/D1 style entries)
        has_courts = False
        for row in rows:
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if cells and (re.search(r"[sd](?:ingles|oubles)?\s*\d", cells[0], re.I)
                          or re.search(r"\b[sd]\d\b", cells[0], re.I)):
                has_courts = True
                break
        if not has_courts:
            continue

        for row in rows:
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 3:
                continue

            court_cell = cells[0].strip().lower()
            singles_m = re.search(r"s(?:ingles)?\s*(\d)", court_cell)
            doubles_m = re.search(r"d(?:oubles)?\s*(\d)", court_cell)
            if singles_m:
                court_num = int(singles_m.group(1))
                court_type = "Singles"
                player_count = 1
            elif doubles_m:
                court_num = int(doubles_m.group(1))
                court_type = "Doubles"
                player_count = 2
            else:
                continue

            court_label = f"Court {court_num} {court_type}"

            # Extract player names from middle cells (between court label and score/result)
            # Typical layouts:
            # [court, home_player, away_player, score, W/L, W/L]  (singles)
            # [court, home_p1, home_p2, away_p1, away_p2, score, W/L, W/L]  (doubles)
            # [court, home_combined, away_combined, score, W/L, W/L]  (doubles combined)
            player_cells = [c for c in cells[1:] if c and not re.match(r"^[\d\-\s]+$", c)
                            and c.upper() not in ("W", "L", "WO", "DEF") and len(c) > 2]

            # Detect W/L from last 2 cells
            home_won: bool | None = None
            for c in reversed(cells):
                cu = c.strip().upper()
                if cu == "W":
                    home_won = True
                    break
                if cu == "L":
                    home_won = False
                    break

            # Assign player cells to home/away
            home_players: list[str] = []
            away_players: list[str] = []
            if len(player_cells) >= 2:
                if player_count == 1:
                    home_players = [_clean_name(player_cells[0])]
                    away_players = [_clean_name(player_cells[1])]
                else:
                    mid = len(player_cells) // 2
                    home_players = [_clean_name(p) for p in player_cells[:mid]]
                    away_players = [_clean_name(p) for p in player_cells[mid:mid + player_count]]

            if is_home:
                our_players = home_players
                their_players = away_players
                we_won = home_won
            else:
                our_players = away_players
                their_players = home_players
                we_won = (not home_won) if home_won is not None else None

            if our_players:
                our_lineup[court_label] = our_players
            if their_players:
                opp_lineup[court_label] = their_players
            if we_won is not None:
                actual_results.append(CourtResult(
                    court=court_num,
                    court_type=court_type,
                    winner="us" if we_won else "them",
                    score="",
                ))

        if our_lineup:  # found a valid court table
            break

    return {
        "date": date_val,
        "home_team": home_team,
        "away_team": away_team,
        "our_lineup": our_lineup,
        "opponent_lineup": opp_lineup,
        "actual_results": actual_results,
    }


def _clean_name(raw: str) -> str:
    """Clean a player name extracted from scorecard HTML."""
    return re.sub(r"\s+", " ", raw).strip()


def _parse_scorecard(html: str) -> list[tuple[tuple[int, str], bool]]:
    """
    Parse a USTA match scorecard page.
    Returns list of ((court_number, court_type), won) for each court.
    court_type is "Singles" or "Doubles".
    'won' is from the perspective of our team (RBW-Long Shots).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[tuple[int, str], bool]] = []
    our_team_fragment = get_my_team_name().lower()

    # USTA scorecards use tables with rows per court
    # Typical columns: Court | Player1 | Player2 | Score | Result
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 3:
                continue

            court_cell = cells[0].strip().lower()
            # Detect court type and number from cell text like "Singles 1", "S1", "Doubles 1", "D1"
            singles_m = re.search(r"s(?:ingles)?\s*(\d)", court_cell)
            doubles_m = re.search(r"d(?:oubles)?\s*(\d)", court_cell)
            if singles_m:
                court_num = int(singles_m.group(1))
                court_type = "Singles"
            elif doubles_m:
                court_num = int(doubles_m.group(1))
                court_type = "Doubles"
            else:
                continue

            # Determine if our team won by finding a "W" or the team name in cells
            row_text = " ".join(cells).lower()
            won = our_team_fragment in row_text and any(
                re.search(r"\bw\b", c.lower()) for c in cells[1:]
            )
            # Fallback: look for explicit "W" in one of the last two cells
            for cell in cells[-2:]:
                c = cell.strip().upper()
                if c == "W":
                    won = True
                    break
                elif c == "L":
                    won = False
                    break

            results.append(((court_num, court_type), won))

    return results


def _parse_roster_html(html: str) -> dict[str, dict]:
    """
    Parse the USTA TennisLink team roster tab.
    Extracts player name, profile URL, WTN singles/doubles, and NTRP from the roster table.
    WTN may be shown directly in the table — if so, no separate profile fetch is needed.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    players: dict[str, dict] = {}

    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        # Roster tables have a "name" or "player" column
        if not any("name" in h or "player" in h for h in headers):
            continue

        name_idx = next((i for i, h in enumerate(headers) if "name" in h or "player" in h), None)
        ntrp_idx = next((i for i, h in enumerate(headers) if "ntrp" in h or "rating" in h), None)
        # WTN columns may be labelled "wtn", "singles wtn", "doubles wtn", "wtn singles", etc.
        wtn_s_idx = next((i for i, h in enumerate(headers)
                          if ("wtn" in h and "single" in h) or h == "wtn singles" or h == "singles wtn"), None)
        wtn_d_idx = next((i for i, h in enumerate(headers)
                          if ("wtn" in h and "double" in h) or h == "wtn doubles" or h == "doubles wtn"), None)
        # Generic "wtn" column — treat as singles if no split columns found
        if wtn_s_idx is None and wtn_d_idx is None:
            wtn_s_idx = next((i for i, h in enumerate(headers) if h == "wtn"), None)

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if not cells or name_idx is None or name_idx >= len(cells):
                continue

            name_cell = cells[name_idx]
            name = name_cell.get_text(strip=True)
            if not name or len(name) < 3:
                continue

            # Extract profile link from name cell
            profile_url = ""
            a = name_cell.find("a", href=True)
            if a:
                href = a["href"]
                profile_url = href if href.startswith("http") else f"https://tennislink.usta.com{href}"

            def _cell_float(idx):
                if idx is None or idx >= len(cells):
                    return None
                try:
                    v = float(cells[idx].get_text(strip=True))
                    return v if v > 0 else None
                except (ValueError, TypeError):
                    return None

            players[name] = {
                "profile_url": profile_url,
                "ntrp": _cell_float(ntrp_idx),
                "wtn_singles": _cell_float(wtn_s_idx),
                "wtn_doubles": _cell_float(wtn_d_idx),
            }

        if players:
            break

    return players


def _parse_scorecard_player_profiles(html: str, our_team_fragment: str) -> dict[str, dict]:
    """
    Extract OPPONENT player names and profile links from a match scorecard page.
    Identifies which side is the opponent by checking team name.
    Returns {player_name: {"profile_url": str, "wtn_singles": None, "wtn_doubles": None}}.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    players: dict[str, dict] = {}

    # Determine which side is ours by scanning headers/labels for team name
    text = soup.get_text(" ").lower()
    # Find "vs" split to identify home/away
    is_home = our_team_fragment[:6] in text[:text.find(" vs ")] if " vs " in text else True

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            court_text = cells[0].get_text(strip=True).lower()
            if not (re.search(r"[sd](?:ingles|oubles)?\s*\d", court_text)
                    or re.search(r"\b[sd]\d\b", court_text)):
                continue

            # Collect all player-name cells with links (cells 1 onward, skip score/result cells)
            linked_players: list[tuple[str, str]] = []
            for cell in cells[1:]:
                for a in cell.find_all("a", href=True):
                    name = a.get_text(strip=True)
                    href = a["href"]
                    if name and len(name) > 3 and not _parse_date_flexible(name):
                        full = href if href.startswith("http") else f"https://tennislink.usta.com{href}"
                        linked_players.append((name, full))

            # The opponent side depends on is_home: away players are second half
            mid = len(linked_players) // 2
            opp_players = linked_players[mid:] if is_home else linked_players[:mid]
            for name, url in opp_players:
                if name not in players:
                    players[name] = {"profile_url": url, "wtn_singles": None, "wtn_doubles": None}

    return players


def _parse_standings(html: str) -> dict:
    """
    Parse the USTA Standings tab HTML.
    Returns our position, total teams, wins/losses, and matches remaining.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    our_team = get_my_team_name().lower()
    standings: list[dict] = []

    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        has_win = any("win" in h for h in headers)
        has_team = any("team" in h for h in headers)
        if not (has_win and has_team):
            continue

        team_idx = next((i for i, h in enumerate(headers) if "team" in h), None)
        win_idx  = next((i for i, h in enumerate(headers) if h == "win" or h == "wins" or h == "w"), None)
        loss_idx = next((i for i, h in enumerate(headers) if h in ("loss", "losses", "l")), None)

        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells or team_idx is None or team_idx >= len(cells):
                continue
            try:
                wins   = int(cells[win_idx])  if win_idx  is not None and win_idx  < len(cells) else 0
                losses = int(cells[loss_idx]) if loss_idx is not None and loss_idx < len(cells) else 0
            except ValueError:
                continue
            standings.append({
                "team":   cells[team_idx],
                "wins":   wins,
                "losses": losses,
                "matches": wins + losses,
            })

        if standings:
            break

    our_entry = next((s for s in standings if our_team in s["team"].lower()), None)
    position  = next((i + 1 for i, s in enumerate(standings) if our_team in s["team"].lower()), None)

    return {
        "standings": standings,
        "our_position": position,
        "total_teams": len(standings),
        "our_wins":   our_entry["wins"]   if our_entry else None,
        "our_losses": our_entry["losses"] if our_entry else None,
        "our_matches_played": our_entry["matches"] if our_entry else None,
    }


def _clean_team(raw: str) -> str:
    """Remove extra whitespace injected by USTA's table rendering."""
    return re.sub(r"\s+", " ", raw).strip()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _click_single_tab(page, tab_id: str) -> None:
    """Click one USTA ASP.NET tab and wait for the postback to complete."""
    try:
        locator = page.locator(f"#{tab_id}")
        locator.wait_for(state="visible", timeout=10_000)
        with page.expect_navigation(wait_until="domcontentloaded", timeout=15_000):
            locator.click()
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
    except Exception:
        pass


def _click_usta_tabs(page) -> None:
    """Legacy: click both tabs (used only by debug_page)."""
    _click_single_tab(page, "ctl00_mainContent_lnkMatchScheduleForTeams")
    _click_single_tab(page, "ctl00_mainContent_lnkMatchSummaryForTeams")


def _click_all_tabs(page, captured: list[dict]) -> None:
    """Legacy debug helper — clicks every visible tab and prints labels."""
    clicked: set[str] = set()
    for sel in ("ul.nav-tabs li a", "ul.nav li a", "a[href*='Schedule']",
                "a[href*='Results']", "a[href*='Standings']", "a[href*='Roster']"):
        try:
            for link in page.locator(sel).all():
                label = (link.inner_text() or "").strip()
                if label and label not in clicked:
                    clicked.add(label)
                    print(f"    Clicking tab: {label}")
                    try:
                        link.click(timeout=3_000)
                        page.wait_for_load_state("networkidle", timeout=5_000)
                    except Exception:
                        pass
        except Exception:
            pass
    if not clicked:
        print("    No tabs found.")


def _extract_team_league_url(captured: list[dict]) -> Optional[str]:
    """
    Pull the team-specific league URL from the MyTennis dashboard response.
    Returns a relative URL like '/Leagues/Main/StatsAndStandings.aspx?t=R-3&par1=...&par2=...'
    or None if not found.
    """
    for item in captured:
        data = item.get("data", {})
        result = data.get("GetMyTennisResult", {})
        events = result.get("MyEvents", {}) if result else {}
        link = events.get("LeaguesLink") if events else None
        if link and "par1=" in link:
            return link
    return None


def _find_list(data, keys: tuple) -> list:
    """Recursively search a JSON structure for a list under any of the given keys."""
    if isinstance(data, list) and data:
        # If the list items look like match rows, return as-is
        if isinstance(data[0], dict):
            return data
    if isinstance(data, dict):
        for k in keys:
            if k in data and isinstance(data[k], list):
                return data[k]
        # recurse one level
        for v in data.values():
            result = _find_list(v, keys)
            if result:
                return result
    return []


def _str(row: dict, *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def _hdr_idx(headers: list[str], *candidates: str) -> Optional[int]:
    for c in candidates:
        for i, h in enumerate(headers):
            if c in h:
                return i
    return None


def _parse_date_flexible(raw: str) -> Optional[datetime.date]:
    if not raw:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m-%d-%Y", "%B %d, %Y"):
        try:
            return datetime.datetime.strptime(raw.strip()[:19], fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _parse_win(win_field, score: str) -> Optional[bool]:
    if win_field is not None:
        if isinstance(win_field, bool):
            return win_field
        s = str(win_field).lower()
        if s in ("true", "win", "w", "1"):
            return True
        if s in ("false", "loss", "l", "0"):
            return False
    # Try to infer from score like "3-2" (first number = our courts won)
    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", score)
    if m:
        return int(m.group(1)) > int(m.group(2))
    return None
