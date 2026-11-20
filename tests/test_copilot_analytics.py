from pathlib import Path

from copilot_analytics import classify_intents, retrieve_analytics


DB_PATH = Path(__file__).parents[1] / "data" / "sciip.db"


def test_semantic_aliases_map_to_business_intents():
    assert "customer_activity" in classify_intents("Which customers are becoming less active?")
    assert "demand_patterns" in classify_intents("What parts have unpredictable demand?")
    assert "replenishment" in classify_intents("Which parts should we prioritize for replenishment?")


def test_retrieval_returns_live_evidence_for_customer_activity():
    evidence, sources, intents = retrieve_analytics(
        "Which customers are becoming less active?", DB_PATH, limit=3
    )
    assert "customer_activity" in intents
    assert "customer_profiles" in sources
    assert evidence["customer_profiles"]


def test_project_scope_is_applied_to_part_demand():
    evidence, _sources, _intents = retrieve_analytics(
        "What parts have unpredictable demand?", DB_PATH, project="ETCH", limit=10
    )
    rows = evidence["part_demand_patterns"]
    assert rows
    assert all(row["project"] == "ETCH" for row in rows)
