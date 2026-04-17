import os
from pathlib import Path
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


def get_groq_api_key() -> str:
    return _require("GROQ_API_KEY")


def get_anthropic_api_key() -> str:
    return _require("ANTHROPIC_API_KEY")


def get_usta_credentials() -> tuple[str, str]:
    return _require("USTA_USERNAME"), _require("USTA_PASSWORD")


def get_ai_provider() -> str:
    return os.getenv("AI_PROVIDER", "groq").lower()


def get_my_team_url() -> str:
    return _require("MY_TEAM_URL")


def get_usta_team_url() -> str:
    return _require("USTA_TEAM_URL")
