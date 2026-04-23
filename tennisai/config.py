import os
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}\n"
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


# ---------------------------------------------------------------------------
# AI provider keys
# ---------------------------------------------------------------------------

def get_groq_api_key() -> str:
    return _require("GROQ_API_KEY")


def get_anthropic_api_key() -> str:
    return _require("ANTHROPIC_API_KEY")


def get_gemini_api_key() -> str:
    return _require("GEMINI_API_KEY")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def get_ai_provider() -> str:
    return os.getenv("AI_PROVIDER", "groq").lower()


# ---------------------------------------------------------------------------
# USTA credentials
# ---------------------------------------------------------------------------

def get_usta_credentials() -> tuple[str, str]:
    return _require("USTA_USERNAME"), _require("USTA_PASSWORD")


# ---------------------------------------------------------------------------
# Multi-team support
# ---------------------------------------------------------------------------

def get_team_count() -> int:
    """Total number of configured teams."""
    try:
        return max(1, int(os.getenv("TEAM_COUNT", "1")))
    except ValueError:
        return 1


def get_active_team_index() -> int:
    """1-based index of the currently active team."""
    try:
        return max(1, int(os.getenv("ACTIVE_TEAM", "1")))
    except ValueError:
        return 1


def get_team_config(team_num: int = 0) -> dict:
    """
    Return config for a team. team_num=0 uses the active team.
    Keys: index, name, team_url, usta_url.
    Falls back to legacy MY_TEAM_* / USTA_TEAM_URL keys for single-team setups.
    """
    if team_num == 0:
        team_num = get_active_team_index()

    # Numbered keys take priority; legacy keys are the fallback for team 1
    name = os.getenv(f"TEAM_{team_num}_NAME") or (
        os.getenv("MY_TEAM_NAME", "") if team_num == 1 else ""
    )
    team_url = os.getenv(f"TEAM_{team_num}_URL") or (
        os.getenv("MY_TEAM_URL", "") if team_num == 1 else ""
    )
    usta_url = os.getenv(f"TEAM_{team_num}_USTA_URL") or (
        os.getenv("USTA_TEAM_URL", "") if team_num == 1 else ""
    )

    return {
        "index": team_num,
        "name": name,
        "team_url": team_url,
        "usta_url": usta_url,
    }


def get_all_teams() -> list[dict]:
    """Return config dicts for all configured teams."""
    return [get_team_config(i) for i in range(1, get_team_count() + 1)]


def set_active_team(index: int) -> None:
    """Persist ACTIVE_TEAM=index to .env and update the current process environment."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        if re.search(r"^ACTIVE_TEAM=", content, re.MULTILINE):
            content = re.sub(r"(?m)^ACTIVE_TEAM=.*", f"ACTIVE_TEAM={index}", content)
        else:
            content += f"\nACTIVE_TEAM={index}\n"
        env_path.write_text(content, encoding="utf-8")
    os.environ["ACTIVE_TEAM"] = str(index)


# ---------------------------------------------------------------------------
# Convenience getters that proxy through the active team
# ---------------------------------------------------------------------------

def get_my_team_url() -> str:
    url = get_team_config()["team_url"]
    if not url:
        raise EnvironmentError(
            "Team URL not configured. "
            "Set TEAM_N_URL (or MY_TEAM_URL for a single-team setup) in .env."
        )
    return url


def get_usta_team_url() -> str:
    url = get_team_config()["usta_url"]
    if not url:
        raise EnvironmentError(
            "USTA URL not configured. "
            "Set TEAM_N_USTA_URL (or USTA_TEAM_URL for a single-team setup) in .env."
        )
    return url


def get_scoring_format() -> str:
    return os.getenv(
        "SCORING_FORMAT",
        "2 sets (tiebreak at 6-6); super tiebreak to 10 if sets split 1-1",
    )


def get_my_team_name() -> str:
    return get_team_config()["name"]


def get_singles_courts_override() -> Optional[int]:
    val = os.getenv("SINGLES_COURTS")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
