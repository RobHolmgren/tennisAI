#!/usr/bin/env python3
"""
TennisAI Integration Test Runner
=================================
Tests the live CLI against real USTA TennisLink and tennisrecord.com endpoints.
Requires a valid .env file with credentials and team URLs.

Usage:
    python tennisai_test.py              # run all test cases
    python tennisai_test.py --tc TC1     # run a single test case
    python tennisai_test.py --tc TC1 TC2 # run specific test cases

Test cases:
    TC1  check-tennisrecord  — validates player/team scraping
    TC2  check-usta          — validates USTA login and schedule fetch
    TC3  suggest-lineup      — validates lineup suggestion (interactive, automated)
    TC4  precheck-config     — validates .env pre-check catches missing fields
    TC5  switch-team         — validates multi-team switching
    TC6  update-players      — validates player file creation
    TC7  list-matches        — validates match persistence after analyze
    TC8  accuracy            — validates accuracy report handles empty state
    TC9  update-wtn          — validates WTN fetch smoke test
"""

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).parent
TIMEOUT_SHORT = 60    # seconds — non-network commands
TIMEOUT_MEDIUM = 120  # seconds — single network call
TIMEOUT_LONG = 300    # seconds — multi-step interactive commands


@dataclass
class TestResult:
    name: str
    passed: bool
    details: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def run(args: list[str], input_text: str = "", timeout: int = TIMEOUT_MEDIUM) -> tuple[int, str, str]:
    """Run `python -m tennisai <args>` and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [PYTHON, "-m", "tennisai"] + args,
        capture_output=True,
        text=True,
        input=input_text,
        timeout=timeout,
        cwd=PROJECT_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


def assert_check(results: list[str], condition: bool, message: str) -> bool:
    status = "  PASS" if condition else "  FAIL"
    results.append(f"{status}  {message}")
    return condition


def skip(name: str, reason: str) -> TestResult:
    return TestResult(name=name, passed=True, skipped=True, skip_reason=reason)


# ---------------------------------------------------------------------------
# TC1 — check-tennisrecord
# ---------------------------------------------------------------------------

def tc1_check_tennisrecord() -> TestResult:
    name = "TC1 check-tennisrecord"
    details = []
    all_passed = True

    try:
        rc, stdout, stderr = run(["check-tennisrecord"], timeout=TIMEOUT_MEDIUM)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  Command timed out"])

    combined = stdout + stderr

    all_passed &= assert_check(details, rc == 0,
        f"Exit code is 0 (got {rc})")
    all_passed &= assert_check(details, "ERROR" not in combined.upper() or "no errors" in combined.lower(),
        "Output contains no ERROR messages")

    player_match = re.search(r"Players found \((\d+)\)", stdout)
    player_count = int(player_match.group(1)) if player_match else 0
    all_passed &= assert_check(details, player_count > 0,
        f"Players found > 0 (got {player_count})")

    team_match = re.search(r"Teams in your league \((\d+)\)", stdout)
    team_count = int(team_match.group(1)) if team_match else 0
    all_passed &= assert_check(details, team_count > 0,
        f"League teams found > 0 (got {team_count})")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC2 — check-usta
# ---------------------------------------------------------------------------

def tc2_check_usta() -> TestResult:
    name = "TC2 check-usta"
    details = []
    all_passed = True

    try:
        rc, stdout, stderr = run(["check-usta"], timeout=TIMEOUT_LONG)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  Command timed out"])

    all_passed &= assert_check(details, rc == 0,
        f"Exit code is 0 (got {rc})")
    all_passed &= assert_check(details, "Login successful" in stdout,
        "Output contains 'Login successful'")
    all_passed &= assert_check(details, "No matches found" not in stdout,
        "Schedule is not empty")

    date_found = bool(re.search(r"\d{4}-\d{2}-\d{2}", stdout))
    all_passed &= assert_check(details, date_found,
        "At least one match date (YYYY-MM-DD) appears in schedule output")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC3 — suggest-lineup (interactive, automated)
# ---------------------------------------------------------------------------

def tc3_suggest_lineup() -> TestResult:
    name = "TC3 suggest-lineup"
    details = []
    all_passed = True

    # Step 1: determine how many upcoming matches exist (reuse TC2 output)
    try:
        rc, usta_out, _ = run(["check-usta"], timeout=TIMEOUT_LONG)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False,
                          details=["FAIL  check-usta timed out — cannot prepare TC3"])
    if rc != 0:
        return TestResult(name=name, passed=False,
                          details=["FAIL  check-usta failed — cannot prepare TC3"])

    date_matches = re.findall(r"\d{4}-\d{2}-\d{2}", usta_out)
    upcoming_count = len(date_matches)
    if upcoming_count < 2:
        return skip(name, f"Only {upcoming_count} upcoming match(es) — need ≥ 2 to test non-first selection")

    # Pick a random match that is NOT the first
    match_num = random.randint(2, upcoming_count)

    # Step 2: determine roster size (reuse TC1 output)
    try:
        rc2, tr_out, _ = run(["check-tennisrecord"], timeout=TIMEOUT_MEDIUM)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False,
                          details=["FAIL  check-tennisrecord timed out — cannot prepare TC3"])

    player_match = re.search(r"Players found \((\d+)\)", tr_out)
    total_players = int(player_match.group(1)) if player_match else 8
    available_count = random.randint(8, max(8, total_players))
    # Build player number list: pick the first available_count players (1-indexed)
    player_nums = " ".join(str(i) for i in range(1, available_count + 1))

    # Capture opponent name from the schedule for later validation
    # "vs <opponent>" from the match listing
    # We need to identify which opponent corresponds to our chosen match number.
    # check-usta output lists matches like "YYYY-MM-DD: Home vs Away @ Location"
    schedule_lines = [l for l in usta_out.splitlines() if re.search(r"\d{4}-\d{2}-\d{2}", l)]
    selected_line = schedule_lines[match_num - 1] if match_num <= len(schedule_lines) else ""

    # Input sequence: match selection → player numbers → accept AI lineup
    stdin_input = f"{match_num}\n{player_nums}\ny\n"

    try:
        rc3, stdout, stderr = run(
            ["suggest-lineup"],
            input_text=stdin_input,
            timeout=TIMEOUT_LONG,
        )
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  suggest-lineup timed out"])

    combined = stdout + stderr

    # Assertion: no errors
    has_error = bool(re.search(r"\bError\b", combined, re.IGNORECASE)) and rc3 != 0
    all_passed &= assert_check(details, not has_error,
        f"No errors in output (exit code {rc3})")

    # Parse which players were declared available from stdout
    available_section_match = re.search(
        r"players selected: (.+)", stdout, re.IGNORECASE
    )
    available_names: set[str] = set()
    if available_section_match:
        available_names = {n.strip() for n in available_section_match.group(1).split(",")}

    # Parse all "Us:" lines to collect assigned player names
    assigned_players: list[str] = []
    for line in stdout.splitlines():
        if line.strip().startswith("Us:"):
            names_part = line.split("Us:", 1)[1].strip()
            for name in names_part.split(","):
                n = name.strip()
                if n and n != "TBD":
                    assigned_players.append(n)

    # Assertion: no player on two courts
    duplicates = {p for p in assigned_players if assigned_players.count(p) > 1}
    all_passed &= assert_check(details, len(duplicates) == 0,
        f"No player assigned to more than one court (duplicates: {duplicates or 'none'})")

    # Assertion: all courts have players
    tbd_count = stdout.count("TBD")
    all_passed &= assert_check(details, tbd_count == 0,
        f"All courts have players assigned (TBD count: {tbd_count})")

    # Assertion: only available players assigned
    if available_names:
        outside = {p for p in assigned_players if p not in available_names}
        all_passed &= assert_check(details, len(outside) == 0,
            f"Only available players assigned (outside available: {outside or 'none'})")

    # Assertion: opponent lineup prediction header mentions selected match's opponent
    opp_header_match = re.search(r"OPPONENT LINEUP PREDICTION:\s*(.+)", stdout)
    selected_opponent = re.search(r"vs\s+(.+?)(?:\s+@|\s*$)", stdout.split("Selected:")[1].split("\n")[0]).group(1).strip() if "Selected:" in stdout else ""
    if opp_header_match and selected_opponent:
        header_opponent = opp_header_match.group(1).strip()
        # Fuzzy match: first 8 chars of opponent name should appear in header
        opponent_key = selected_opponent[:8].lower()
        match_ok = opponent_key in header_opponent.lower() or header_opponent.lower() in selected_opponent.lower()
        all_passed &= assert_check(details, match_ok,
            f"Opponent in prediction header ('{header_opponent}') matches selected match opponent ('{selected_opponent}')")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC4 — precheck catches missing config
# ---------------------------------------------------------------------------

def tc4_precheck_config() -> TestResult:
    name = "TC4 precheck-config"
    details = []
    all_passed = True

    # Run with a stripped environment that has no USTA_USERNAME
    env = os.environ.copy()
    env.pop("USTA_USERNAME", None)
    env["USTA_USERNAME"] = ""

    try:
        result = subprocess.run(
            [PYTHON, "-m", "tennisai", "list-teams"],
            capture_output=True, text=True, env=env,
            timeout=TIMEOUT_SHORT, cwd=PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  Command timed out"])

    rc, stdout, stderr = result.returncode, result.stdout, result.stderr
    combined = stdout + stderr

    all_passed &= assert_check(details, rc != 0,
        f"Exit code is non-zero when USTA_USERNAME is missing (got {rc})")
    all_passed &= assert_check(details, "USTA_USERNAME" in combined,
        "Error output names the missing field 'USTA_USERNAME'")
    all_passed &= assert_check(details, ".env" in combined.lower(),
        "Error output mentions .env")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC5 — switch-team
# ---------------------------------------------------------------------------

def tc5_switch_team() -> TestResult:
    name = "TC5 switch-team"
    details = []
    all_passed = True

    # Check how many teams are configured
    try:
        rc, stdout, _ = run(["list-teams"], timeout=TIMEOUT_SHORT)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  list-teams timed out"])

    all_passed &= assert_check(details, rc == 0, f"list-teams exit code 0 (got {rc})")

    team_lines = [l for l in stdout.splitlines() if re.search(r"\d+\.", l)]
    team_count = len(team_lines)
    all_passed &= assert_check(details, team_count >= 1,
        f"At least 1 team listed (found {team_count})")

    # Find current active team
    active_match = re.search(r"\[\*\]\s*(\d+)\.", stdout)
    original_team = int(active_match.group(1)) if active_match else 1

    if team_count < 2:
        details.append("  SKIP  Only 1 team configured — switch test skipped")
        return TestResult(name=name, passed=all_passed, details=details)

    # Switch to a different team
    target = 2 if original_team == 1 else 1
    try:
        rc2, stdout2, _ = run(["switch-team", str(target)], timeout=TIMEOUT_SHORT)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  switch-team timed out"])

    all_passed &= assert_check(details, rc2 == 0, f"switch-team {target} exit code 0")
    all_passed &= assert_check(details, f"Switched to team {target}" in stdout2,
        f"Output confirms switch to team {target}")

    # Verify list-teams now shows target as active
    try:
        rc3, stdout3, _ = run(["list-teams"], timeout=TIMEOUT_SHORT)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  list-teams timed out"])

    active_now = re.search(r"\[\*\]\s*(\d+)\.", stdout3)
    new_active = int(active_now.group(1)) if active_now else -1
    all_passed &= assert_check(details, new_active == target,
        f"list-teams shows team {target} as active (got {new_active})")

    # Restore original team
    run(["switch-team", str(original_team)], timeout=TIMEOUT_SHORT)
    details.append(f"  INFO  Restored active team to {original_team}")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC6 — update-players creates files
# ---------------------------------------------------------------------------

def tc6_update_players() -> TestResult:
    name = "TC6 update-players"
    details = []
    all_passed = True

    try:
        rc, stdout, stderr = run(["update-players"], timeout=TIMEOUT_LONG)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  Command timed out"])

    combined = stdout + stderr

    all_passed &= assert_check(details, rc == 0, f"Exit code is 0 (got {rc})")
    all_passed &= assert_check(details, "Traceback" not in combined,
        "No Python traceback in output")

    players_dir = PROJECT_ROOT / "players"
    player_files = list(players_dir.glob("*.json")) if players_dir.exists() else []
    all_passed &= assert_check(details, len(player_files) > 0,
        f"Player JSON files exist in players/ (found {len(player_files)})")

    created_match = re.search(r"created (\d+) new", stdout)
    updated_match = re.search(r"updated (\d+) existing", stdout)
    total_reported = (
        (int(created_match.group(1)) if created_match else 0) +
        (int(updated_match.group(1)) if updated_match else 0)
    )
    all_passed &= assert_check(details, total_reported > 0,
        f"At least 1 player created or updated (reported: {total_reported})")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC7 — list-matches (smoke test: shows saved matches or correct empty message)
# ---------------------------------------------------------------------------

def tc7_list_matches() -> TestResult:
    name = "TC7 list-matches"
    details = []
    all_passed = True

    try:
        rc, stdout, stderr = run(["list-matches"], timeout=TIMEOUT_SHORT)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  Command timed out"])

    combined = stdout + stderr

    all_passed &= assert_check(details, rc == 0, f"Exit code is 0 (got {rc})")
    all_passed &= assert_check(details, "Traceback" not in combined,
        "No Python traceback in output")

    has_matches = bool(re.search(r"[0-9a-f]{8}", stdout))  # match IDs are 8-char hex
    has_empty_msg = "no matches" in stdout.lower() or "run 'analyze'" in stdout.lower()
    all_passed &= assert_check(details, has_matches or has_empty_msg,
        "Output shows saved matches or appropriate empty-state message")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC8 — accuracy command handles empty state gracefully
# ---------------------------------------------------------------------------

def tc8_accuracy() -> TestResult:
    name = "TC8 accuracy"
    details = []
    all_passed = True

    try:
        rc, stdout, stderr = run(["accuracy"], timeout=TIMEOUT_SHORT)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  Command timed out"])

    combined = stdout + stderr

    all_passed &= assert_check(details, rc == 0, f"Exit code is 0 (got {rc})")
    all_passed &= assert_check(details, "Traceback" not in combined,
        "No Python traceback in output")

    # Either shows stats or a graceful "no data" message
    has_stats = "PREDICTION ACCURACY" in stdout
    has_empty_msg = "no completed" in stdout.lower() or "record-result" in stdout.lower()
    all_passed &= assert_check(details, has_stats or has_empty_msg,
        "Output shows accuracy stats or graceful empty-state message")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# TC9 — update-wtn smoke test
# ---------------------------------------------------------------------------

def tc9_update_wtn() -> TestResult:
    name = "TC9 update-wtn"
    details = []
    all_passed = True

    # update-wtn requires existing player files; skip if none exist
    players_dir = PROJECT_ROOT / "players"
    if not players_dir.exists() or not list(players_dir.glob("*.json")):
        return skip(name, "No player files found — run TC6 (update-players) first")

    # Pipe "n" to decline manual entry for any players not found
    try:
        rc, stdout, stderr = run(["update-wtn"], input_text="n\n", timeout=TIMEOUT_LONG)
    except subprocess.TimeoutExpired:
        return TestResult(name=name, passed=False, details=["FAIL  Command timed out"])

    combined = stdout + stderr

    all_passed &= assert_check(details, rc == 0, f"Exit code is 0 (got {rc})")
    all_passed &= assert_check(details, "Traceback" not in combined,
        "No Python traceback in output")
    all_passed &= assert_check(details, "Done." in stdout or "player(s)" in stdout,
        "Output contains completion message")

    return TestResult(name=name, passed=all_passed, details=details)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TCS: dict[str, callable] = {
    "TC1": tc1_check_tennisrecord,
    "TC2": tc2_check_usta,
    "TC3": tc3_suggest_lineup,
    "TC4": tc4_precheck_config,
    "TC5": tc5_switch_team,
    "TC6": tc6_update_players,
    "TC7": tc7_list_matches,
    "TC8": tc8_accuracy,
    "TC9": tc9_update_wtn,
}

COL = 42  # label column width


def _colour(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def _print_result(r: TestResult) -> None:
    if r.skipped:
        label = _colour("SKIP", "33")
        print(f"  [{label}] {r.name.ljust(COL)}  {r.skip_reason}")
    elif r.passed:
        label = _colour("PASS", "32")
        print(f"  [{label}] {r.name}")
    else:
        label = _colour("FAIL", "31")
        print(f"  [{label}] {r.name}")
    for line in r.details:
        print(f"         {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TennisAI integration test runner")
    parser.add_argument("--tc", nargs="+", metavar="TCN",
                        help="Run specific test case(s), e.g. --tc TC1 TC2")
    args = parser.parse_args()

    to_run = args.tc if args.tc else list(ALL_TCS.keys())
    unknown = [t for t in to_run if t.upper() not in ALL_TCS]
    if unknown:
        print(f"Unknown test case(s): {unknown}. Valid: {list(ALL_TCS.keys())}")
        sys.exit(1)

    print(f"\nTennisAI Integration Tests")
    print("=" * 60)

    results: list[TestResult] = []
    for tc_name in to_run:
        fn = ALL_TCS[tc_name.upper()]
        print(f"\nRunning {tc_name.upper()}...")
        t0 = time.time()
        try:
            result = fn()
        except Exception as exc:
            result = TestResult(name=tc_name.upper(), passed=False,
                                details=[f"FAIL  Unexpected exception: {exc}"])
        elapsed = time.time() - t0
        _print_result(result)
        print(f"         ({elapsed:.1f}s)")
        results.append(result)

    print("\n" + "=" * 60)
    passed  = sum(1 for r in results if r.passed and not r.skipped)
    failed  = sum(1 for r in results if not r.passed)
    skipped = sum(1 for r in results if r.skipped)
    total   = len(results)

    summary_parts = [f"{_colour(str(passed), '32')} passed"]
    if failed:
        summary_parts.append(f"{_colour(str(failed), '31')} failed")
    if skipped:
        summary_parts.append(f"{_colour(str(skipped), '33')} skipped")
    summary_parts.append(f"{total} total")

    print("Result: " + ", ".join(summary_parts))
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
