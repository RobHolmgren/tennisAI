"""
Pre-run environment validation.
Called before every CLI command to catch misconfigured .env files early.
"""
import os
import sys
from pathlib import Path


def validate_env() -> None:
    """
    Check that all required .env fields are present and non-empty.
    Prints every missing field at once, then exits with a helpful message.
    Skips validation when running --help.
    """
    errors: list[str] = []

    def _missing(key: str) -> bool:
        val = os.getenv(key, "")
        return not val or val.strip() == ""

    # Core credentials
    if _missing("USTA_USERNAME"):
        errors.append("  USTA_USERNAME — your USTA TennisLink login email")
    if _missing("USTA_PASSWORD"):
        errors.append("  USTA_PASSWORD — your USTA TennisLink password")

    # AI provider
    provider = os.getenv("AI_PROVIDER", "").lower().strip()
    if not provider:
        errors.append("  AI_PROVIDER — set to one of: ollama, groq, gemini, claude")
    elif provider not in ("ollama", "groq", "gemini", "claude"):
        errors.append(
            f"  AI_PROVIDER={provider!r} is not recognised. "
            "Valid values: ollama, groq, gemini, claude"
        )

    # Team URLs — support both multi-team (TEAM_N_*) and legacy (MY_TEAM_*) formats
    try:
        team_count = max(1, int(os.getenv("TEAM_COUNT", "1")))
    except ValueError:
        team_count = 1

    if team_count > 1:
        for n in range(1, team_count + 1):
            if _missing(f"TEAM_{n}_NAME"):
                errors.append(f"  TEAM_{n}_NAME — name for team {n}")
            if _missing(f"TEAM_{n}_URL"):
                errors.append(f"  TEAM_{n}_URL — tennisrecord.com URL for team {n}")
            if _missing(f"TEAM_{n}_USTA_URL"):
                errors.append(f"  TEAM_{n}_USTA_URL — USTA TennisLink URL for team {n}")
    else:
        # Single-team: accept either TEAM_1_* or legacy MY_TEAM_* keys
        has_name = not _missing("TEAM_1_NAME") or not _missing("MY_TEAM_NAME")
        has_url = not _missing("TEAM_1_URL") or not _missing("MY_TEAM_URL")
        has_usta = not _missing("TEAM_1_USTA_URL") or not _missing("USTA_TEAM_URL")
        if not has_name:
            errors.append("  MY_TEAM_NAME (or TEAM_1_NAME) — your team name")
        if not has_url:
            errors.append("  MY_TEAM_URL (or TEAM_1_URL) — your tennisrecord.com team URL")
        if not has_usta:
            errors.append(
                "  USTA_TEAM_URL (or TEAM_1_USTA_URL) — your USTA TennisLink Stats & Standings URL"
            )

    if errors:
        print("\nConfiguration error — the following required fields are missing or empty in .env:")
        for line in errors:
            print(line)
        print(
            "\nCopy .env.example to .env and fill in the missing values, then try again.\n"
            "  cp .env.example .env"
        )
        sys.exit(1)
