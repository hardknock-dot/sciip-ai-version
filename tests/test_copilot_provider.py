import json
import sqlite3
from pathlib import Path

import copilot


def _db_path(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "sciip.db"
    sqlite3.connect(db_path).close()
    return db_path


def test_ui_saved_provider_selection_precedes_environment_default(tmp_path, monkeypatch):
    db_path = _db_path(tmp_path)
    (db_path.parent / "copilot_config.json").write_text(json.dumps({
        "name": "Groq",
        "api_key": "saved-key",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    }), encoding="utf-8")
    monkeypatch.setenv("SCIIP_AI_PROVIDER_1", "gemini")
    monkeypatch.setenv("SCIIP_AI_API_KEY_1", "environment-key")
    monkeypatch.setenv("SCIIP_AI_MODEL_1", "gemini-3.8-flash")
    monkeypatch.setenv("SCIIP_AI_BASE_URL_1", "https://generativelanguage.googleapis.com/v1beta/openai/")
    for slot in (2, 3, 4):
        monkeypatch.delenv(f"SCIIP_AI_PROVIDER_{slot}", raising=False)
        monkeypatch.delenv(f"SCIIP_AI_API_KEY_{slot}", raising=False)
        monkeypatch.delenv(f"SCIIP_AI_MODEL_{slot}", raising=False)
        monkeypatch.delenv(f"SCIIP_AI_BASE_URL_{slot}", raising=False)

    config = copilot.load_active_provider_config(db_path)

    assert config.name == "Groq"
    assert config.api_key == "saved-key"
    assert config.model == "llama-3.3-70b-versatile"


def test_environment_configuration_is_used_when_no_ui_selection_exists(tmp_path, monkeypatch):
    db_path = _db_path(tmp_path)
    monkeypatch.setenv("SCIIP_AI_PROVIDER_1", "gemini")
    monkeypatch.setenv("SCIIP_AI_API_KEY_1", "environment-key")
    monkeypatch.setenv("SCIIP_AI_MODEL_1", "gemini-3.8-flash")

    config = copilot.load_active_provider_config(db_path)

    assert config.name == "Google Gemini"
    assert config.api_key == "environment-key"
    assert config.model == "gemini-3.8-flash"


def test_selected_provider_uses_its_matching_env_slot(tmp_path, monkeypatch):
    db_path = _db_path(tmp_path)
    (db_path.parent / "copilot_config.json").write_text(json.dumps({
        "name": "Groq",
        "api_key": "old-key",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }), encoding="utf-8")
    monkeypatch.setenv("SCIIP_AI_PROVIDER_2", "groq")
    monkeypatch.setenv("SCIIP_AI_API_KEY_2", "groq-env-key")
    monkeypatch.setenv("SCIIP_AI_MODEL_2", "llama-3.1-8b-instant")
    monkeypatch.setenv("SCIIP_AI_BASE_URL_2", "https://api.groq.com/openai/v1")

    config = copilot.load_active_provider_config(db_path)

    assert config.name == "Groq"
    assert config.api_key == "groq-env-key"
    assert config.model == "llama-3.1-8b-instant"
    assert config.base_url == "https://api.groq.com/openai/v1"


def test_empty_saved_configuration_does_not_block_environment(tmp_path, monkeypatch):
    db_path = _db_path(tmp_path)
    (db_path.parent / "copilot_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SCIIP_AI_PROVIDER_1", "openai")
    monkeypatch.setenv("SCIIP_AI_API_KEY_1", "environment-key")
    monkeypatch.setenv("SCIIP_AI_MODEL_1", "gpt-4o-mini")
    monkeypatch.delenv("SCIIP_AI_BASE_URL_1", raising=False)

    config = copilot.load_active_provider_config(db_path)

    assert config.name == "OpenAI"
    assert config.api_key == "environment-key"
    assert config.base_url == "https://api.openai.com/v1"


def test_provider_error_classification():
    class RateLimitError(Exception):
        status_code = 429

    class NotFoundError(Exception):
        status_code = 404

    assert copilot.classify_provider_error(RateLimitError())[0:2] == ("rate_limit", True)
    assert copilot.classify_provider_error(NotFoundError())[0:2] == ("model_unavailable", False)


def test_health_check_requires_credentials():
    result = copilot.check_provider_health(copilot.ProviderConfig(
        name="Google Gemini",
        api_key="",
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ))

    assert result.ok is False
    assert result.category == "missing_credentials"


def test_retrieval_reads_existing_database_schema():
    db_path = Path(__file__).parents[1] / "data" / "sciip.db"
    context, sources = copilot.retrieve_context("Which customers are becoming less active?", db_path)

    assert "TABLE customer_profiles" in context
    assert "customer_profiles" in sources
