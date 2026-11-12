from pathlib import Path
from backend.triage import TriageEngine

ROOT = Path(__file__).resolve().parents[1]
engine = TriageEngine(ROOT / "data" / "maintenance_tickets.csv")


def test_gearbox_ticket_is_critical_and_routed():
    result = engine.process(
        "Gearbox vibration has increased with abnormal noise", "wind"
    )
    assert result["classification"]["category"] == "gearbox_vibration"
    assert result["severity"] == "critical"
    assert result["routing"]["priority_queue"] == "P1 - Immediate"
    assert len(result["agent_trace"]) == 5


def test_smoke_escalates_to_critical():
    result = engine.process("Smoke coming from the inverter enclosure", "solar")
    assert result["severity"] == "critical"
    assert result["routing"]["priority_queue"].startswith("P1")


def test_cosmetic_issue_uses_general_maintenance():
    result = engine.process(
        "Minor paint damage and faded safety label on the inverter cabinet",
        "solar",
    )
    assert result["classification"]["category"] == "general_maintenance"
    assert result["severity"] == "low"
    assert result["routing"]["team"] == "Central Maintenance Triage"
    assert result["routing"]["priority_queue"] == "P4 - Low"


def test_unknown_issue_uses_safe_fallback():
    result = engine.process("Please inspect this unusual asset condition", "unknown")
    assert result["classification"]["category"] == "general_maintenance"
    assert result["human_review_required"] is True
