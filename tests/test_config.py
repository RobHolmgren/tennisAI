"""
Unit tests for tennisai/config.py and tennisai/precheck.py.
No network calls — all values are injected via environment variables.
"""
import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(**kwargs):
    """Return a dict of env vars with only the provided keys set."""
    base = {
        "USTA_USERNAME": "user@example.com",
        "USTA_PASSWORD": "secret",
        "AI_PROVIDER": "ollama",
        "MY_TEAM_NAME": "Test Team",
        "MY_TEAM_URL": "https://tennisrecord.com/test",
        "USTA_TEAM_URL": "https://tennislink.usta.com/test",
    }
    base.update(kwargs)
    # Allow callers to remove a key by passing None
    return {k: v for k, v in base.items() if v is not None}


# ---------------------------------------------------------------------------
# precheck.validate_env
# ---------------------------------------------------------------------------

class TestValidateEnv:

    def test_valid_single_team_config_passes(self):
        with patch.dict(os.environ, _env(), clear=True):
            from tennisai.precheck import validate_env
            validate_env()  # should not raise

    def test_missing_usta_username_fails(self):
        env = _env(USTA_USERNAME=None)
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_missing_usta_password_fails(self):
        env = _env(USTA_PASSWORD=None)
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_missing_ai_provider_fails(self):
        env = _env(AI_PROVIDER=None)
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_invalid_ai_provider_fails(self):
        env = _env(AI_PROVIDER="openai")
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_all_valid_ai_providers_pass(self):
        from tennisai.precheck import validate_env
        for provider in ("ollama", "groq", "gemini", "claude"):
            with patch.dict(os.environ, _env(AI_PROVIDER=provider), clear=True):
                validate_env()  # should not raise

    def test_missing_team_name_fails(self):
        env = _env(MY_TEAM_NAME=None)
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_missing_team_url_fails(self):
        env = _env(MY_TEAM_URL=None)
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_missing_usta_url_fails(self):
        env = _env(USTA_TEAM_URL=None)
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_team1_keys_accepted_as_alternative_to_legacy(self):
        env = {
            "USTA_USERNAME": "u", "USTA_PASSWORD": "p", "AI_PROVIDER": "ollama",
            "TEAM_1_NAME": "T1", "TEAM_1_URL": "https://tr.com", "TEAM_1_USTA_URL": "https://usta.com",
        }
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            validate_env()

    def test_multi_team_missing_team2_url_fails(self):
        env = {
            "USTA_USERNAME": "u", "USTA_PASSWORD": "p", "AI_PROVIDER": "ollama",
            "TEAM_COUNT": "2",
            "TEAM_1_NAME": "T1", "TEAM_1_URL": "https://tr.com/1", "TEAM_1_USTA_URL": "https://usta.com/1",
            "TEAM_2_NAME": "T2",
            # TEAM_2_URL and TEAM_2_USTA_URL intentionally missing
        }
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()

    def test_multi_team_all_present_passes(self):
        env = {
            "USTA_USERNAME": "u", "USTA_PASSWORD": "p", "AI_PROVIDER": "ollama",
            "TEAM_COUNT": "2",
            "TEAM_1_NAME": "T1", "TEAM_1_URL": "https://tr.com/1", "TEAM_1_USTA_URL": "https://usta.com/1",
            "TEAM_2_NAME": "T2", "TEAM_2_URL": "https://tr.com/2", "TEAM_2_USTA_URL": "https://usta.com/2",
        }
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            validate_env()

    def test_error_message_lists_all_missing_fields(self, capsys):
        env = _env(USTA_USERNAME=None, USTA_PASSWORD=None, MY_TEAM_NAME=None)
        with patch.dict(os.environ, env, clear=True):
            from tennisai.precheck import validate_env
            with pytest.raises(SystemExit):
                validate_env()
        out = capsys.readouterr().out
        assert "USTA_USERNAME" in out
        assert "USTA_PASSWORD" in out
        assert "MY_TEAM_NAME" in out or "TEAM_1_NAME" in out


# ---------------------------------------------------------------------------
# config.get_team_config — backward compatibility
# ---------------------------------------------------------------------------

class TestGetTeamConfig:

    def test_legacy_keys_read_for_team1(self):
        # Explicitly blank out the numbered keys so the legacy fallback is exercised.
        # load_dotenv() runs at import time and may pre-populate TEAM_1_NAME in
        # os.environ; overriding with "" ensures the `or` fallback fires.
        env = {
            "MY_TEAM_NAME": "Legacy Team",
            "MY_TEAM_URL": "https://tr.com/legacy",
            "USTA_TEAM_URL": "https://usta.com/legacy",
            "TEAM_1_NAME": "",
            "TEAM_1_URL": "",
            "TEAM_1_USTA_URL": "",
            "TEAM_COUNT": "1",
        }
        with patch.dict(os.environ, env):
            from tennisai.config import get_team_config
            cfg = get_team_config(1)
            assert cfg["name"] == "Legacy Team"
            assert cfg["team_url"] == "https://tr.com/legacy"
            assert cfg["usta_url"] == "https://usta.com/legacy"

    def test_numbered_keys_take_priority_over_legacy(self):
        env = {
            "MY_TEAM_NAME": "Old Name",
            "MY_TEAM_URL": "https://tr.com/old",
            "USTA_TEAM_URL": "https://usta.com/old",
            "TEAM_1_NAME": "New Name",
            "TEAM_1_URL": "https://tr.com/new",
            "TEAM_1_USTA_URL": "https://usta.com/new",
        }
        with patch.dict(os.environ, env, clear=True):
            from tennisai.config import get_team_config
            cfg = get_team_config(1)
            assert cfg["name"] == "New Name"
            assert cfg["team_url"] == "https://tr.com/new"

    def test_team2_has_no_legacy_fallback(self):
        env = {
            "MY_TEAM_NAME": "Team A",
            "TEAM_COUNT": "2",
            # No TEAM_2_NAME
        }
        with patch.dict(os.environ, env, clear=True):
            from tennisai.config import get_team_config
            cfg = get_team_config(2)
            assert cfg["name"] == ""

    def test_team_count_defaults_to_1(self):
        with patch.dict(os.environ, {}, clear=True):
            from tennisai.config import get_team_count
            assert get_team_count() == 1

    def test_team_count_reads_env(self):
        with patch.dict(os.environ, {"TEAM_COUNT": "3"}, clear=True):
            from tennisai.config import get_team_count
            assert get_team_count() == 3

    def test_invalid_team_count_falls_back_to_1(self):
        with patch.dict(os.environ, {"TEAM_COUNT": "notanumber"}, clear=True):
            from tennisai.config import get_team_count
            assert get_team_count() == 1

    def test_active_team_defaults_to_1(self):
        with patch.dict(os.environ, {}, clear=True):
            from tennisai.config import get_active_team_index
            assert get_active_team_index() == 1

    def test_get_all_teams_returns_correct_count(self):
        env = {
            "TEAM_COUNT": "2",
            "TEAM_1_NAME": "T1", "TEAM_1_URL": "u1", "TEAM_1_USTA_URL": "u1",
            "TEAM_2_NAME": "T2", "TEAM_2_URL": "u2", "TEAM_2_USTA_URL": "u2",
        }
        with patch.dict(os.environ, env, clear=True):
            from tennisai.config import get_all_teams
            teams = get_all_teams()
            assert len(teams) == 2
            assert teams[0]["name"] == "T1"
            assert teams[1]["name"] == "T2"
