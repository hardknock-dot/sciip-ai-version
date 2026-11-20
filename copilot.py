"""SCIIP AI Copilot: Grounded Retrieval, Multi-Provider Switching, Persistent History, and Streamlit Tab UI.

The Copilot uses the existing SCIIP SQLite database as its source of truth.
Retrieval is grounded in live tables (domain intent, entity scoring, bounded queries).
Conversation history is permanently stored in data/copilot_conversations.json.
API provider settings (Gemini, OpenAI, Groq, OpenRouter, Mistral, Ollama, Offline) can be changed dynamically.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from copilot_analytics import retrieve_analytics
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load default .env if present
load_dotenv(Path(__file__).with_name(".env"))

logger = logging.getLogger("sciip.copilot")

MAX_ROWS_PER_TABLE = 18
MAX_CONTEXT_CHARS = 28_000
SUPPORTED_TABLES = {
    "customer_profiles", "customer_yearly_trend", "part_customer_breakdown",
    "part_demand_patterns", "part_forecast_accuracy", "part_forecast_horizon",
    "part_forecasts", "part_intelligence_summary", "part_monthly_demand",
    "part_yearly_trend", "project_intelligence", "replenishment_recommendations",
    "risk_alerts", "demand_lines", "item_master", "part_master",
    "customer_forecast_decomposition",
}

DOMAIN_SIGNALS = {
    "customer_activity": ("customer_profiles", "part_customer_breakdown", "customer_yearly_trend", "demand_lines"),
    "overdue": ("customer_profiles", "demand_lines", "part_customer_breakdown", "risk_alerts"),
    "demand_pattern": ("part_demand_patterns", "part_intelligence_summary", "part_monthly_demand", "part_forecast_accuracy"),
    "trend": ("part_yearly_trend", "customer_yearly_trend", "part_monthly_demand", "part_intelligence_summary"),
    "forecast": ("part_forecasts", "part_forecast_horizon", "part_forecast_accuracy", "part_intelligence_summary"),
    "inventory": ("replenishment_recommendations", "item_master", "part_master", "risk_alerts"),
    "risk": ("risk_alerts", "replenishment_recommendations", "part_intelligence_summary", "project_intelligence"),
    "orders": ("demand_lines", "customer_profiles", "part_customer_breakdown"),
    "project": ("project_intelligence", "part_intelligence_summary", "risk_alerts"),
}

SIGNAL_TERMS = {
    "customer_activity": {"customer", "customers", "active", "activity", "quiet", "churn", "engagement", "ordering", "recency"},
    "overdue": {"overdue", "late", "past", "due", "open", "backlog", "outstanding"},
    "demand_pattern": {"unpredictable", "erratic", "lumpy", "intermittent", "volatile", "variability", "pattern", "demand"},
    "trend": {"trend", "trends", "growing", "declining", "decreasing", "increasing", "history", "historical"},
    "forecast": {"forecast", "predict", "prediction", "future", "expected", "probability"},
    "inventory": {"inventory", "stock", "replenishment", "replenish", "supply", "coverage", "purchase", "procure"},
    "risk": {"risk", "alert", "alerts", "critical", "shortage", "priority"},
    "orders": {"orders", "order", "sales", "shipment", "shipments", "quantity", "qty"},
    "project": {"project", "etch", "lam", "customer", "business"},
}

PROVIDER_PRESETS = {
    "Google Gemini": {
        "id": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "default_model": "gemini-3.8-flash",
        "env_var": "SCIIP_AI_API_KEY_1",
    },
    "OpenAI": {
        "id": "openai",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "o1-mini"],
        "default_model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "Groq": {
        "id": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "default_model": "llama-3.3-70b-versatile",
        "env_var": "GROQ_API_KEY",
    },
    "OpenRouter": {
        "id": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct"],
        "default_model": "openai/gpt-4o-mini",
        "env_var": "OPENROUTER_API_KEY",
    },
    "Mistral AI": {
        "id": "mistral",
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "mistral-large-latest"],
        "default_model": "mistral-small-latest",
        "env_var": "MISTRAL_API_KEY",
    },
    "Together AI": {
        "id": "together",
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "env_var": "TOGETHER_API_KEY",
    },
    "Local Ollama / Custom API": {
        "id": "ollama",
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.2", "mistral", "qwen2.5", "deepseek-r1"],
        "default_model": "llama3.2",
        "env_var": "",
    },
    "Offline Grounded Database": {
        "id": "offline",
        "base_url": "",
        "models": ["sqlite-direct-evidence"],
        "default_model": "sqlite-direct-evidence",
        "env_var": "",
    },
}


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    model: str
    base_url: str | None = None


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    model: str
    ok: bool
    category: str
    detail: str
    latency_ms: int | None = None


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _provider_id(name: str) -> str:
    value = (name or "").strip().lower()
    for display_name, preset in PROVIDER_PRESETS.items():
        if value in {display_name.lower(), preset["id"].lower()}:
            return preset["id"]
    return value


def _provider_display_name(provider_id: str) -> str:
    provider_id = _provider_id(provider_id)
    for display_name, preset in PROVIDER_PRESETS.items():
        if preset["id"] == provider_id:
            return display_name
    return provider_id or "Google Gemini"


def _provider_defaults(provider_id: str) -> tuple[str, str]:
    preset = PROVIDER_PRESETS.get(_provider_display_name(provider_id), PROVIDER_PRESETS["Google Gemini"])
    return preset["base_url"], preset["default_model"]


def _normalise_provider(name: str, api_key: str, model: str, base_url: str | None) -> ProviderConfig:
    provider_id = _provider_id(name or "gemini")
    display_name = _provider_display_name(provider_id)
    default_url, default_model = _provider_defaults(provider_id)
    return ProviderConfig(
        name=display_name,
        api_key=(api_key or "").strip(),
        model=(model or default_model).strip(),
        base_url=(base_url or default_url or None),
    )


def _saved_base_url(provider_name: str, base_url: str | None) -> str | None:
    """Repair a saved config when it contains another preset's default URL."""
    if not base_url:
        return None
    known_urls = {
        preset["base_url"]
        for preset in PROVIDER_PRESETS.values()
        if preset.get("base_url")
    }
    provider = PROVIDER_PRESETS.get(_provider_display_name(provider_name))
    if provider and base_url in known_urls:
        return provider.get("base_url") or None
    return base_url


def classify_provider_error(exc: Exception) -> tuple[str, bool, str]:
    """Classify provider failures without returning credentials or raw payloads."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    name = type(exc).__name__.lower()
    if status == 401 or "authentication" in name:
        return "authentication", False, "API key rejected"
    if status == 403 or "permission" in name:
        return "permission", False, "Provider access denied"
    if status == 404 or "notfound" in name:
        return "model_unavailable", False, "Model or endpoint unavailable"
    if status == 429 or "rate" in name or "quota" in str(exc).lower():
        return "rate_limit", True, "Rate limit or quota exceeded"
    if status in {500, 502, 503, 504} or "server" in name:
        return "provider_unavailable", True, "Provider temporarily unavailable"
    if "timeout" in name or "timed out" in str(exc).lower():
        return "timeout", True, "Provider request timed out"
    return "provider_error", False, "Provider request failed"


# ---------------------------------------------------------------------------
# Persistent Provider Configuration
# ---------------------------------------------------------------------------
def _get_config_path(db_path: str | Path) -> Path:
    return Path(db_path).parent / "copilot_config.json"


def load_active_provider_config(db_path: str | Path) -> ProviderConfig:
    """Load the provider explicitly selected in the UI, then use env defaults."""
    cfg_file = _get_config_path(db_path)
    if cfg_file.exists():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if data and isinstance(data, dict) and (data.get("name") or data.get("api_key")):
                saved = _normalise_provider(
                    data.get("name", "Google Gemini"),
                    data.get("api_key", ""),
                    data.get("model", ""),
                    _saved_base_url(data.get("name", "Google Gemini"), data.get("base_url") or None),
                )
                # If this provider is defined in .env, its slot is the source
                # of truth for that provider's credential/model/endpoint. The
                # saved JSON still supplies the user's selected provider.
                env_config = _configured_provider_from_env(saved.name)
                if env_config:
                    return env_config
                return saved
        except Exception as e:
            logger.warning("Error reading copilot_config.json: %s", e)

    env_name = _env("SCIIP_AI_PROVIDER_1")
    env_key = _env("SCIIP_AI_API_KEY_1") or _env("OPENAI_API_KEY")
    env_model = _env("SCIIP_AI_MODEL_1")
    env_url = _env("SCIIP_AI_BASE_URL_1")
    return _normalise_provider(env_name or "gemini", env_key, env_model, env_url)


def _configured_key_for_provider(provider_name: str) -> str:
    """Resolve a provider's key from its fallback slot or conventional env name."""
    provider_id = _provider_id(provider_name)
    for config in configured_fallback_providers():
        if _provider_id(config.name) == provider_id and config.api_key:
            return config.api_key
    preset = PROVIDER_PRESETS.get(_provider_display_name(provider_id), {})
    return _env(preset.get("env_var", "")) if preset.get("env_var") else ""


def _configured_provider_from_env(provider_name: str) -> ProviderConfig | None:
    """Return the matching environment slot for a provider, if configured."""
    provider_id = _provider_id(provider_name)
    for config in configured_fallback_providers():
        if _provider_id(config.name) == provider_id:
            return config
    return None


def save_active_provider_config(db_path: str | Path, config: ProviderConfig) -> bool:
    cfg_file = _get_config_path(db_path)
    try:
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(config)
        cfg_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        logger.error("Failed to save active provider config: %s", e)
        return False


def configured_fallback_providers() -> list[ProviderConfig]:
    """Return fallback providers from environment variables."""
    providers: list[ProviderConfig] = []
    for i in range(1, 5):
        name = _env(f"SCIIP_AI_PROVIDER_{i}")
        key = _env(f"SCIIP_AI_API_KEY_{i}")
        if not name and i == 1 and _env("OPENAI_API_KEY"):
            name, key = "openai", _env("OPENAI_API_KEY")
        if not name or (not key and _provider_id(name) != "ollama"):
            continue
        providers.append(_normalise_provider(
            name,
            key,
            _env(f"SCIIP_AI_MODEL_{i}"),
            _env(f"SCIIP_AI_BASE_URL_{i}") or None,
        ))
    return providers


def check_provider_health(config: ProviderConfig, timeout: float = 8.0) -> ProviderHealth:
    """Make a bounded, minimal provider request for an actual connection status."""
    provider_id = _provider_id(config.name)
    if provider_id == "offline":
        return ProviderHealth(config.name, config.model, True, "offline", "Local database evidence only")
    if not config.api_key and provider_id != "ollama":
        return ProviderHealth(config.name, config.model, False, "missing_credentials", "API key is not configured")
    started = time.perf_counter()
    try:
        client = OpenAI(
            api_key=config.api_key or "ollama",
            base_url=config.base_url,
            timeout=timeout,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Reply with OK only."}],
            temperature=0,
            max_tokens=5,
        )
        ok = bool(response.choices and response.choices[0].message.content)
        return ProviderHealth(
            config.name,
            config.model,
            ok,
            "healthy" if ok else "empty_response",
            "Connection successful" if ok else "Provider returned no text",
            int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        category, _retryable, detail = classify_provider_error(exc)
        return ProviderHealth(
            config.name,
            config.model,
            False,
            category,
            detail,
            int((time.perf_counter() - started) * 1000),
        )


# ---------------------------------------------------------------------------
# Persistent Multi-Session Conversation Store
# ---------------------------------------------------------------------------
def _get_history_file(db_path: str | Path) -> Path:
    return Path(db_path).parent / "copilot_conversations.json"


def _generate_session_id() -> str:
    return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def load_conversation_store(db_path: str | Path) -> dict[str, Any]:
    hist_file = _get_history_file(db_path)
    if hist_file.exists():
        try:
            data = json.loads(hist_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "sessions" in data:
                # Deduplicate sessions by ID and clean up any identical keys
                seen_ids = set()
                unique_sessions = []
                for s in data.get("sessions", []):
                    sid = s.get("id")
                    if sid and sid not in seen_ids:
                        seen_ids.add(sid)
                        unique_sessions.append(s)
                data["sessions"] = unique_sessions
                if not data["sessions"]:
                    new_id = _generate_session_id()
                    data = {
                        "active_session_id": new_id,
                        "sessions": [{
                            "id": new_id,
                            "title": "New Investigation",
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                            "messages": []
                        }]
                    }
                return data
        except Exception as e:
            logger.warning("Error reading conversation history: %s", e)
    # Default blank store
    init_id = _generate_session_id()
    return {
        "active_session_id": init_id,
        "sessions": [
            {
                "id": init_id,
                "title": "New Investigation",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "messages": []
            }
        ]
    }


def save_conversation_store(db_path: str | Path, data: dict[str, Any]) -> bool:
    hist_file = _get_history_file(db_path)
    try:
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        hist_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        logger.error("Failed to save conversation store: %s", e)
        return False


def create_new_session(db_path: str | Path, initial_title: str = "New Investigation") -> str:
    store = load_conversation_store(db_path)
    new_id = _generate_session_id()
    new_sess = {
        "id": new_id,
        "title": initial_title,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }
    store["sessions"].insert(0, new_sess)
    store["active_session_id"] = new_id
    save_conversation_store(db_path, store)
    return new_id


def delete_session(db_path: str | Path, session_id: str) -> None:
    store = load_conversation_store(db_path)
    store["sessions"] = [s for s in store.get("sessions", []) if s.get("id") != session_id]
    if not store["sessions"]:
        create_new_session(db_path)
    else:
        if store.get("active_session_id") == session_id:
            store["active_session_id"] = store["sessions"][0]["id"]
        save_conversation_store(db_path, store)


def clear_all_history(db_path: str | Path) -> None:
    init_id = _generate_session_id()
    store = {
        "active_session_id": init_id,
        "sessions": [
            {
                "id": init_id,
                "title": "New Investigation",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "messages": []
            }
        ]
    }
    save_conversation_store(db_path, store)


# ---------------------------------------------------------------------------
# Grounded Retrieval Engine
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_\-]+", text.lower()))


def _entities(question: str, conn: sqlite3.Connection) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {"part_code": set(), "cust_name": set(), "project": set()}
    q = question.lower()
    for table, column in (("part_intelligence_summary", "part_code"), ("customer_profiles", "cust_name"), ("project_intelligence", "project")):
        try:
            values = pd.read_sql_query(f'SELECT DISTINCT "{column}" FROM "{table}"', conn)[column].dropna().astype(str)
        except Exception:
            continue
        result[column] = {value for value in values if value.lower() in q}
    return result


def _score_row(row: pd.Series, question_tokens: set[str], entity_values: dict[str, set[str]]) -> float:
    text = " ".join(str(v) for v in row.to_dict().values() if pd.notna(v)).lower()
    score = len(question_tokens & _tokens(text)) * 0.15
    for column, values in entity_values.items():
        if values and str(row.get(column, "")) in values:
            score += 8
    return score


def retrieve_context(
    question: str,
    db_path: str | Path,
    dashboard_context: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """Retrieve evidence from the live SQLite database."""
    dashboard_context = dashboard_context or {}
    recent_history = "\n".join(
        str(item.get("content", ""))[:500]
        for item in (history or [])[-4:]
        if item.get("role") in {"user", "assistant"}
    )
    retrieval_question = f"{recent_history}\n{question}" if recent_history else question
    entity_conn = sqlite3.connect(db_path)
    try:
        entities = _entities(retrieval_question, entity_conn)
    finally:
        entity_conn.close()
    scope_project = dashboard_context.get("project")
    if (not scope_project or str(scope_project).lower() == "all") and len(entities.get("project", set())) == 1:
        scope_project = next(iter(entities["project"]))
    analytics_evidence, analytics_sources, _intents = retrieve_analytics(
        retrieval_question,
        db_path,
        project=scope_project,
        entities=entities,
        limit=MAX_ROWS_PER_TABLE,
    )
    if analytics_evidence:
        chunks = [
            f"TABLE {table}\n{json.dumps(rows, default=str)}"
            for table, rows in analytics_evidence.items()
        ]
        return "\n\n".join(chunks)[:MAX_CONTEXT_CHARS], analytics_sources

    # Compatibility fallback for questions that do not map to a known intent.
    conn = sqlite3.connect(db_path)
    try:
        q_tokens = _tokens(question)
        scores = {signal: len(q_tokens & terms) for signal, terms in SIGNAL_TERMS.items()}
        active = [signal for signal, score in scores.items() if score > 0]
        if not active:
            active = ["trend", "risk", "orders"]
        tables: list[str] = []
        for signal in sorted(active, key=lambda s: scores[s], reverse=True):
            for table in DOMAIN_SIGNALS[signal]:
                if table not in tables:
                    tables.append(table)
        entities = _entities(question, conn)
        chunks: list[str] = []
        sources: list[str] = []
        for table in tables[:8]:
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            df["_retrieval_score"] = df.apply(lambda row: _score_row(row, q_tokens, entities), axis=1)
            preferred = [c for c in ("severity", "supply_risk", "status", "recency_days", "cv2", "forecast_next_6mo_qty") if c in df]
            if preferred:
                df = df.sort_values(["_retrieval_score", *preferred], ascending=[False] + [False] * len(preferred), na_position="last")
            else:
                df = df.sort_values("_retrieval_score", ascending=False)
            selected = df.head(MAX_ROWS_PER_TABLE).drop(columns="_retrieval_score")
            chunks.append(f"TABLE {table}\n{selected.to_json(orient='records', date_format='iso')}")
            sources.append(table)
        context = "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]
        return context or "No matching records were found in the SCIIP database.", sources
    finally:
        conn.close()


def _local_grounded_answer(question: str, context: str, sources: list[str]) -> str:
    if not context or context.startswith("No matching"):
        return "I couldn't find supporting records in the SCIIP database for that question."
    return (
        "### 📊 Offline Database Grounded Analysis\n\n"
        "Here are the grounded operational records retrieved directly from the SCIIP database:\n\n"
        "```json\n" + context[:9000] + "\n```\n\n"
        f"**Supporting Tables:** {', '.join(sources)}\n\n"
        "_Tip: Select an active AI Provider in the Settings above to receive full natural language synthesis._"
    )


# ---------------------------------------------------------------------------
# Answer Generation with Active Provider & Fallback
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    db_path: str | Path,
    history: list[dict[str, Any]] | None = None,
    provider_override: ProviderConfig | None = None,
    dashboard_context: dict[str, Any] | None = None,
) -> tuple[str, list[str], str | None]:
    """Synthesize grounded answer using the selected provider or fallback order."""
    context, sources = retrieve_context(question, db_path, dashboard_context, history)

    # Check if user explicitly set Offline mode
    active = provider_override or load_active_provider_config(db_path)
    if active.name.lower().startswith("offline") or active.api_key == "offline":
        return _local_grounded_answer(question, context, sources), sources, "Offline Database"

    system = (
        "You are SCIIP AI Copilot, a senior semiconductor supply chain and inventory operations analyst. "
        "Answer questions accurately using only the supplied SCIIP database context. "
        "Explain inventory positions, demand volatility (Smooth, Erratic, Intermittent, Lumpy), "
        "customer ordering cadence, forecast projections, and replenishment priorities clearly. "
        "Never hallucinate numbers. Cite supporting evidence inline like [table:column/value]. "
        "If data is missing, note it plainly. Separate observed facts from recommendations. "
        "For analytical questions, organize the response as: Executive summary, Key findings, "
        "Metrics, Implications, Recommended actions, and Evidence. Keep it concise and actionable.\n\n"
        f"GROUNDED DATABASE CONTEXT:\n{context}"
    )

    messages = [{"role": "system", "content": system}]
    for item in (history or [])[-6:]:
        if item.get("role") in {"user", "assistant"}:
            prior = re.sub(r"```.*?```", "[structured evidence omitted]", str(item["content"]), flags=re.S)
            messages.append({"role": item["role"], "content": prior[:1800]})
    messages.append({"role": "user", "content": question})

    # Provider attempt queue: selected provider first, then configured fallbacks.
    providers_queue: list[ProviderConfig] = []
    if active.api_key or _provider_id(active.name) == "ollama":
        providers_queue.append(active)
    for p in configured_fallback_providers():
        signature = (_provider_id(p.name), p.base_url or "", p.model)
        existing = {(_provider_id(x.name), x.base_url or "", x.model) for x in providers_queue}
        if signature not in existing:
            providers_queue.append(p)

    if not providers_queue:
        return _local_grounded_answer(question, context, sources), sources, None

    failures: list[tuple[str, str]] = []
    for prov in providers_queue:
        for attempt in range(2):
            try:
                client = OpenAI(
                    api_key=prov.api_key or "ollama",
                    base_url=prov.base_url,
                    timeout=25,
                    max_retries=0,
                )
                response = client.chat.completions.create(
                    model=prov.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000,
                )
                text = response.choices[0].message.content if response.choices else ""
                if text:
                    logger.info("SCIIP Copilot answered using: %s (%s)", prov.name, prov.model)
                    return text, sources, prov.name
                failures.append((prov.name, "empty_response"))
                break
            except Exception as exc:
                category, retryable, detail = classify_provider_error(exc)
                failures.append((prov.name, category))
                logger.warning(
                    "SCIIP provider failed: provider=%s model=%s category=%s attempt=%d detail=%s",
                    prov.name, prov.model, category, attempt + 1, detail,
                )
                if retryable and attempt == 0:
                    time.sleep(0.5)
                    continue
                break

    fallback = _local_grounded_answer(question, context, sources)
    failure_summary = "; ".join(f"{name}: {category}" for name, category in failures) or "no provider configured"
    return (
        "⚠️ **AI providers were unavailable.**\n\n"
        f"Attempt summary: `{failure_summary}`\n\n"
        "Here is the grounded SCIIP database evidence directly:\n\n" + fallback,
        sources,
        None
    )


# ---------------------------------------------------------------------------
# Dedicated Full-Page AI Copilot Module Tab
# ---------------------------------------------------------------------------
def render_copilot_tab(db_path: str | Path, dashboard_context: dict[str, Any] | None = None) -> None:
    """Full-page interactive AI Copilot tab with persistent chat history and provider settings."""
    active_prov = load_active_provider_config(db_path)
    store = load_conversation_store(db_path)
    sessions = store.get("sessions", [])

    # Active session validation
    active_session_id = st.session_state.get("copilot_tab_session_id") or store.get("active_session_id")
    current_session = next((s for s in sessions if s["id"] == active_session_id), None)
    if not current_session and sessions:
        current_session = sessions[0]
        active_session_id = current_session["id"]

    # -----------------------------------------------------------------------
    # Top Action & Provider Bar
    # -----------------------------------------------------------------------
    st.markdown('<div class="scp-panel" style="padding:18px 22px; margin-bottom:18px;">', unsafe_allow_html=True)
    tb_col1, tb_col2 = st.columns([2.5, 1.2])
    with tb_col1:
        st.markdown(
            f'<div style="font-size:20px; font-weight:700; color:#F8FAFC; display:flex; align-items:center; gap:8px;">'
            f'<span>🤖 SCIIP AI Copilot</span> '
            f'<span class="scp-badge-tag" style="margin-left:6px;">INTELLIGENCE ANALYST</span></div>'
            f'<div style="font-size:12px; color:#94A3B8; margin-top:2px;">'
            f'Grounded conversational assistant answering from live semiconductor inventory, demand, and risk tables.</div>',
            unsafe_allow_html=True
        )
    with tb_col2:
        prov_badge_color = "#10B981" if active_prov.api_key else "#F59E0B"
        st.markdown(
            f'<div style="display:flex; flex-direction:column; align-items:flex-end; gap:4px;">'
            f'<div class="scp-pill" style="border-color:{prov_badge_color}; color:{prov_badge_color};">'
            f'⚡ {active_prov.name} &middot; {active_prov.model}</div>'
            f'<div style="font-size:11px; color:#64748B;">{len(sessions)} Saved Conversations</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # API Provider & Model Settings Expander
    # -----------------------------------------------------------------------
    with st.expander("⚙️ API Provider & Model Configuration (Click to Change Provider)", expanded=False):
        st.markdown('<div class="scp-hint">Choose which LLM provider generates responses. You can use official cloud APIs, private proxies, local Ollama, or offline database grounding.</div>', unsafe_allow_html=True)
        p_col1, p_col2, p_col3 = st.columns([1.2, 1.2, 1.6])

        preset_names = list(PROVIDER_PRESETS.keys())
        current_idx = preset_names.index(active_prov.name) if active_prov.name in preset_names else 0
        with p_col1:
            sel_provider_name = st.selectbox("API Provider", preset_names, index=current_idx, key="copilot_cfg_prov")
            selected_preset = PROVIDER_PRESETS[sel_provider_name]

        with p_col2:
            model_options = selected_preset["models"]
            current_model_idx = model_options.index(active_prov.model) if active_prov.model in model_options else 0
            sel_model = st.selectbox("Model Name", model_options + ["Custom..."], index=current_model_idx, key="copilot_cfg_model")
            if sel_model == "Custom...":
                sel_model = st.text_input("Enter Custom Model ID", value=active_prov.model, key="copilot_cfg_custom_model")

        with p_col3:
            # Preload API key from current config, preset env, or blank
            preset_env_val = _configured_key_for_provider(sel_provider_name)
            default_key = active_prov.api_key if sel_provider_name == active_prov.name else preset_env_val
            sel_key = st.text_input("API Key", value=default_key, type="password", key="copilot_cfg_key",
                                    placeholder="Enter provider API key...",
                                    disabled=sel_provider_name == "Offline Grounded Database")

        p_col4, p_col5 = st.columns([2.5, 1])
        with p_col4:
            default_url = selected_preset["base_url"] or ""
            url_val = active_prov.base_url if (sel_provider_name == active_prov.name and active_prov.base_url) else default_url
            sel_url = st.text_input("Base URL (OpenAI-compatible Endpoint)", value=url_val or "", key="copilot_cfg_url",
                                    placeholder="Leave default or enter custom proxy endpoint...",
                                    disabled=sel_provider_name == "Offline Grounded Database")
        with p_col5:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("💾 Save & Apply Provider", type="primary", use_container_width=True, key="copilot_save_prov"):
                new_cfg = ProviderConfig(
                    name=sel_provider_name,
                    api_key=sel_key.strip(),
                    model=sel_model.strip(),
                    base_url=sel_url.strip() if sel_url.strip() else None
                )
                save_active_provider_config(db_path, new_cfg)
                st.toast(f"Switched provider to {sel_provider_name} ({sel_model})!", icon="✅")
                st.rerun()
            if st.button("Test connection", use_container_width=True, key="copilot_test_provider"):
                test_cfg = ProviderConfig(
                    name=sel_provider_name,
                    api_key=sel_key.strip(),
                    model=sel_model.strip(),
                    base_url=sel_url.strip() if sel_url.strip() else None,
                )
                health = check_provider_health(test_cfg)
                st.session_state.copilot_provider_health = asdict(health)
            health_result = st.session_state.get("copilot_provider_health")
            if health_result:
                if health_result.get("ok"):
                    st.success(f"Connected · {health_result.get('latency_ms')} ms")
                else:
                    st.error(f"{health_result.get('category')}: {health_result.get('detail')}")

    # -----------------------------------------------------------------------
    # Main Body: Conversation History Sidebar + Active Chat Stream
    # -----------------------------------------------------------------------
    col_history, col_chat = st.columns([1, 2.7])

    # Left: Conversation History List
    with col_history:
        st.markdown('<div class="scp-panel" style="margin-top:0;">', unsafe_allow_html=True)
        if st.button("➕ New Conversation", type="primary", use_container_width=True, key="copilot_new_chat"):
            new_id = create_new_session(db_path)
            st.session_state.copilot_tab_session_id = new_id
            st.rerun()

        st.markdown('<div class="scp-section-label" style="margin-top:16px;">Previous Conversations</div>', unsafe_allow_html=True)
        if not sessions:
            st.caption("No conversations recorded yet.")
        else:
            for idx, s in enumerate(sessions):
                is_active = (s["id"] == active_session_id)
                msg_count = len(s.get("messages", []))
                
                # Format timestamp
                try:
                    ts_dt = datetime.fromisoformat(s.get("updated_at", s.get("created_at", "")))
                    ts_label = ts_dt.strftime("%d-%b %H:%M")
                except Exception:
                    ts_label = "Recent"

                title = s.get("title", "Investigation")
                display_title = (title[:30] + "…") if len(title) > 32 else title
                
                border_color = "rgba(6, 182, 212, 0.45)" if is_active else "rgba(255, 255, 255, 0.08)"
                bg_color = "rgba(6, 182, 212, 0.12)" if is_active else "rgba(255, 255, 255, 0.02)"

                # Clickable item
                btn_col, del_col = st.columns([4, 1])
                with btn_col:
                    if st.button(
                        f"💬 {display_title}",
                        key=f"conv_sel_{s['id']}_{idx}",
                        use_container_width=True,
                        type="secondary" if not is_active else "primary",
                        help=f"{title}\n{msg_count} messages &middot; {ts_label}"
                    ):
                        st.session_state.copilot_tab_session_id = s["id"]
                        store["active_session_id"] = s["id"]
                        save_conversation_store(db_path, store)
                        st.rerun()
                with del_col:
                    if st.button("✕", key=f"conv_del_{s['id']}_{idx}", help="Delete this conversation"):
                        delete_session(db_path, s["id"])
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All History", key="copilot_clear_all", use_container_width=True):
            clear_all_history(db_path)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Right: Active Chat Stream
    with col_chat:
        st.markdown('<div class="scp-panel" style="margin-top:0; min-height:540px;">', unsafe_allow_html=True)
        if current_session:
            curr_title = current_session.get("title", "Investigation")
            messages = current_session.get("messages", [])

            # Chat Stream Header
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:12px; margin-bottom:16px;">'
                f'<div><span style="font-weight:700; font-size:16px; color:#F8FAFC;">{curr_title}</span> '
                f'<span class="scp-meta-chip" style="margin-left:8px;">{len(messages)} messages</span></div>'
                f'<div style="font-size:11px; color:#94A3B8;">Provider: <span style="color:#38BDF8; font-weight:600;">{active_prov.name}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Empty conversation view with suggested prompt chips
            if not messages:
                st.markdown(
                    '<div class="scp-empty-state" style="padding:40px 20px;">'
                    '<div class="scp-empty-icon">💡</div>'
                    '<div class="scp-empty-title">What would you like to investigate today?</div>'
                    '<div class="scp-empty-desc">Ask anything about customer purchasing health, erratic part demand, replenishment urgencies, or sales forecast reconciliation.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                suggestion_questions = [
                    "Which customers are becoming less active or quiet?",
                    "What parts exhibit lumpy or erratic demand patterns?",
                    "Which parts should we prioritize for replenishment now?",
                    "Are there customers with overdue orders in the selected program?",
                ]
                st.markdown('<div style="font-size:12px; color:#94A3B8; margin:16px 0 8px 0; font-weight:600;">SUGGESTED INVESTIGATIONS:</div>', unsafe_allow_html=True)
                s_cols = st.columns(2)
                for idx, q_text in enumerate(suggestion_questions):
                    with s_cols[idx % 2]:
                        if st.button(f"👉 {q_text}", key=f"tab_sug_{idx}", use_container_width=True):
                            st.session_state["copilot_tab_pending_query"] = q_text
                            st.rerun()

            # Render message stream
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        st.caption(f"📁 Evidence Tables: {', '.join(msg['sources'])}" + (f" &middot; Provider: {msg.get('provider')}" if msg.get('provider') else ""))

            # Handle user input (via chat_input or suggested button click)
            pending_query = st.session_state.pop("copilot_tab_pending_query", None)
            user_query = st.chat_input("Ask SCIIP Copilot about demand, parts, customers, or forecasts...") or pending_query

            if user_query:
                # 1. Update session title if first message
                if not messages:
                    clean_title = re.sub(r"[^\w\s-]", "", user_query).strip()
                    current_session["title"] = (clean_title[:32] + "…") if len(clean_title) > 32 else clean_title

                # 2. Append User Message
                current_session["messages"].append({"role": "user", "content": user_query})
                current_session["updated_at"] = datetime.now().isoformat()
                save_conversation_store(db_path, store)

                # Render user message immediately
                with st.chat_message("user"):
                    st.markdown(user_query)

                # 3. Generate Answer
                with st.chat_message("assistant"):
                    with st.spinner(f"Analyzing SCIIP database using {active_prov.name}…"):
                        ans_text, sources, prov_used = answer_question(
                            user_query,
                            db_path,
                            current_session["messages"][:-1],
                            provider_override=active_prov,
                            dashboard_context=dashboard_context,
                        )
                    st.markdown(ans_text)
                    if sources:
                        st.caption(f"📁 Evidence Tables: {', '.join(sources)}" + (f" &middot; Provider: {prov_used}" if prov_used else ""))

                # 4. Save Assistant Message & Persist
                current_session["messages"].append({
                    "role": "assistant",
                    "content": ans_text,
                    "sources": sources,
                    "provider": prov_used
                })
                current_session["updated_at"] = datetime.now().isoformat()
                save_conversation_store(db_path, store)
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Floating Copilot Drawer Dialog (Synchronized with Persistent Store)
# ---------------------------------------------------------------------------
@st.dialog("SCIIP AI Copilot", width="small")
def _render_copilot_dialog(db_path: str | Path, dashboard_context: dict[str, Any] | None = None) -> None:
    """Render floating drawer dialog synchronized with the persistent conversation store."""
    store = load_conversation_store(db_path)
    active_prov = load_active_provider_config(db_path)
    active_id = store.get("active_session_id")
    sessions = store.get("sessions", [])
    current = next((s for s in sessions if s["id"] == active_id), sessions[0] if sessions else None)

    st.markdown(
        f"<div class='scp-drawer-header'>"
        f"<div><div class='scp-drawer-title'>SCIIP AI Copilot</div>"
        f"<div class='scp-drawer-subtitle'>Provider: {active_prov.name} &middot; {active_prov.model}</div></div>"
        f"<div class='scp-drawer-status'>GROUNDED</div></div>",
        unsafe_allow_html=True
    )

    close_col, clear_col = st.columns([1, 1])
    if close_col.button("Close Drawer", key="copilot_close", use_container_width=True):
        st.session_state.copilot_open = False
        st.rerun()
    if clear_col.button("New Chat", key="copilot_clear", use_container_width=True):
        create_new_session(db_path)
        st.rerun()

    if not current or not current.get("messages"):
        st.markdown(
            "<div class='scp-drawer-empty'>"
            "<div class='scp-drawer-empty-title'>What would you like to investigate?</div>"
            "<div class='scp-drawer-empty-copy'>Use a suggested question or ask about active customers, erratic parts, or supply risks.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    if current:
        for msg in current.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption("Evidence: " + ", ".join(msg["sources"]))

    suggestion_questions = [
        "Which customers are becoming less active?",
        "What parts have unpredictable demand?",
        "Which parts should we prioritize for replenishment?",
    ]
    pending_question = st.session_state.pop("copilot_pending", None)
    if current and not current.get("messages"):
        for index, question_text in enumerate(suggestion_questions):
            if st.button(question_text, key=f"copilot_suggest_{index}", use_container_width=True):
                pending_question = question_text

    input_col, send_col = st.columns([5, 1])
    with input_col:
        question_input = st.text_input("Message", label_visibility="collapsed", placeholder="Ask SCIIP Copilot…", key="copilot_input")
    with send_col:
        send = st.button("Send", key="copilot_send", type="primary", use_container_width=True)

    question = pending_question or (question_input if send else None)
    if question and current:
        if not current.get("messages"):
            clean_title = re.sub(r"[^\w\s-]", "", question).strip()
            current["title"] = (clean_title[:30] + "…") if len(clean_title) > 30 else clean_title

        current["messages"].append({"role": "user", "content": question})
        current["updated_at"] = datetime.now().isoformat()
        save_conversation_store(db_path, store)

        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, sources, provider = answer_question(
                    question,
                    db_path,
                    current["messages"][:-1],
                    provider_override=active_prov,
                    dashboard_context=dashboard_context,
                )
            st.markdown(answer)
            if sources:
                st.caption("Evidence: " + ", ".join(sources) + (f" &middot; Provider: {provider}" if provider else ""))

        current["messages"].append({"role": "assistant", "content": answer, "sources": sources, "provider": provider})
        current["updated_at"] = datetime.now().isoformat()
        save_conversation_store(db_path, store)
        st.rerun()


def render_copilot(db_path: str | Path, dashboard_context: dict[str, Any] | None = None) -> None:
    """Render the fixed launcher button; the chat itself opens as a dialog."""
    if "copilot_open" not in st.session_state:
        st.session_state.copilot_open = False
    with st.container():
        st.markdown("<span class='scp-launcher-anchor'></span>", unsafe_allow_html=True)
        if st.button("✦ AI Copilot", key="copilot_launcher", help="Open SCIIP AI Copilot"):
            st.session_state.copilot_open = not st.session_state.copilot_open
    if st.session_state.copilot_open:
        _render_copilot_dialog(db_path, dashboard_context)
