"""Read-only, parameterized analytical queries for SCIIP Copilot."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


INTENT_ALIASES = {
    "customer_activity": ("less active", "becoming quiet", "went quiet", "churn", "attrition", "recency", "customer activity", "ordering cadence"),
    "overdue_orders": ("overdue order", "late order", "past due", "open order", "backlog", "outstanding order"),
    "demand_patterns": ("unpredictable demand", "erratic demand", "lumpy demand", "intermittent demand", "demand variability", "demand pattern"),
    "forecast": ("forecast", "predicted", "prediction", "expected demand", "future demand", "forecast accuracy"),
    "inventory": ("inventory risk", "stockout", "shortage", "low stock", "supply risk", "inventory"),
    "replenishment": ("replenishment", "replenish", "purchase priority", "procurement priority", "what should we buy"),
    "project": ("project performance", "program performance", "by project", "by program", "project level"),
    "relationships": ("customer part", "customer-part", "customer relationship", "who buys", "top customer"),
    "historical_demand": ("historical demand", "demand history", "monthly demand", "demand trend", "demand trends", "year over year"),
}

INTENT_TABLES = {
    "customer_activity": ("customer_profiles", "customer_yearly_trend"),
    "overdue_orders": ("part_customer_breakdown", "demand_lines", "risk_alerts"),
    "demand_patterns": ("part_demand_patterns", "part_intelligence_summary"),
    "forecast": ("part_forecasts", "part_forecast_horizon", "part_forecast_accuracy"),
    "inventory": ("replenishment_recommendations", "item_master", "part_master"),
    "replenishment": ("replenishment_recommendations", "risk_alerts"),
    "project": ("project_intelligence", "risk_alerts"),
    "relationships": ("part_customer_breakdown", "part_customer_concentration"),
    "historical_demand": ("part_monthly_demand", "part_yearly_trend", "customer_yearly_trend"),
}


def classify_intents(question: str) -> list[str]:
    """Classify semantic business intents using phrase aliases plus token overlap."""
    text = re.sub(r"[^a-z0-9\s-]", " ", question.lower())
    tokens = set(text.split())
    scored: list[tuple[str, int]] = []
    for intent, aliases in INTENT_ALIASES.items():
        score = 0
        for alias in aliases:
            alias_tokens = set(alias.split())
            if alias in text:
                score += 3
            elif alias_tokens & tokens:
                score += 1
        if score:
            scored.append((intent, score))
    if not scored:
        return ["historical_demand", "inventory", "project"]
    return [intent for intent, _score in sorted(scored, key=lambda item: item[1], reverse=True)[:4]]


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _query(conn: sqlite3.Connection, table: str, where: list[str], params: list[Any], order_by: str, limit: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    sql = f'SELECT * FROM "{table}"'
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order_by} LIMIT ?"
    rows = conn.execute(sql, [*params, max(1, min(int(limit), 50))]).fetchall()
    return [dict(row) for row in rows]


def _scope(where: list[str], params: list[Any], project: str | None, column: str = "project") -> None:
    if project and project.lower() != "all":
        where.append(f'"{column}" = ?')
        params.append(project)


def _entity_filter(where: list[str], params: list[Any], entities: dict[str, set[str]], column: str) -> None:
    values = sorted(entities.get(column, set()))
    if values:
        placeholders = ",".join("?" for _ in values)
        where.append(f'"{column}" IN ({placeholders})')
        params.extend(values)


def retrieve_analytics(
    question: str,
    db_path: str | Path,
    project: str | None = None,
    entities: dict[str, set[str]] | None = None,
    limit: int = 18,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[str]]:
    """Run exact read-only queries selected by semantic business intent."""
    entities = entities or {}
    intents = classify_intents(question)
    evidence: dict[str, list[dict[str, Any]]] = {}
    sources: list[str] = []
    with _connect(db_path) as conn:
        for intent in intents:
            where: list[str] = []
            params: list[Any] = []
            table = None
            order_by = "rowid DESC"
            if intent == "customer_activity":
                table, order_by = "customer_profiles", "recency_days DESC"
                if project and project.lower() != "all":
                    where.append('"cust_name" IN (SELECT DISTINCT "cust_name" FROM "demand_lines" WHERE "project" = ?)')
                    params.append(project)
                _entity_filter(where, params, entities, "cust_name")
            elif intent == "overdue_orders":
                table, order_by = "part_customer_breakdown", "days_since_last_order DESC"
                where.append('"relationship_status" = ?')
                params.append("Overdue / went quiet")
                _scope(where, params, project)
                _entity_filter(where, params, entities, "cust_name")
            elif intent == "demand_patterns":
                table, order_by = "part_demand_patterns", "cv2 DESC"
                _scope(where, params, project)
                _entity_filter(where, params, entities, "part_code")
            elif intent == "forecast":
                table, order_by = "part_forecasts", "forecast_next_6mo_qty DESC"
                _scope(where, params, project)
                _entity_filter(where, params, entities, "part_code")
            elif intent in {"inventory", "replenishment"}:
                table, order_by = "replenishment_recommendations", "recommended_qty DESC"
                if project and project.lower() != "all":
                    where.append("(',' || replace(COALESCE(\"projects\", ''), ' ', '') || ',') LIKE ?")
                    params.append(f"%,{project.replace(' ', '')},%")
                _entity_filter(where, params, entities, "part_code")
                if intent == "replenishment":
                    where.append('"status" IN (?, ?)')
                    params.extend(["Replenish Now", "Replenish Soon"])
            elif intent == "project":
                table, order_by = "project_intelligence", "total_revenue DESC"
                _scope(where, params, project)
            elif intent == "relationships":
                table, order_by = "part_customer_breakdown", "cust_qty DESC"
                _scope(where, params, project)
                _entity_filter(where, params, entities, "part_code")
                _entity_filter(where, params, entities, "cust_name")
            elif intent == "historical_demand":
                table, order_by = "part_monthly_demand", "month DESC, qty DESC"
                _scope(where, params, project)
                _entity_filter(where, params, entities, "part_code")
            if not table:
                continue
            rows = _query(conn, table, where, params, order_by, limit)
            if rows:
                evidence[table] = rows
                if table not in sources:
                    sources.append(table)
    return evidence, sources, intents
