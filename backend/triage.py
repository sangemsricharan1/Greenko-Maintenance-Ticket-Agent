"""Explainable ticket triage engine and small-data evaluation utilities."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}
VALID_ASSETS = {"solar", "wind", "hydro", "grid", "unknown"}

CATEGORY_RULES = [
    ("gearbox_vibration", ["gearbox", "vibration", "abnormal noise", "bearing"]),
    ("inverter_thermal", ["inverter", "overheat", "high temperature", "thermal", "tripping"]),
    ("inspection_compliance", ["inspection", "checklist", "spillway", "maintenance schedule"]),
    ("data_gap", ["scada", "missing data", "data missing", "telemetry", "communication loss"]),
    ("pitch_system", ["blade pitch", "pitch error", "pitch system"]),
    ("grid_curtailment", ["curtailment", "generation dashboard", "dispatch instruction", "grid instruction"]),
]

ROOT_CAUSES = {
    "gearbox_vibration": "Possible gearbox bearing wear, lubrication issue, or mechanical misalignment.",
    "inverter_thermal": "Possible inverter cooling obstruction, fan failure, overload, or ambient-temperature effect.",
    "inspection_compliance": "Inspection activity may be overdue, unassigned, or awaiting checklist completion.",
    "data_gap": "Possible SCADA communication interruption, gateway issue, or data pipeline delay.",
    "pitch_system": "Possible pitch actuator, sensor, control-system, or gust-related fault.",
    "grid_curtailment": "Possible dispatch-data integration, mapping, or dashboard synchronization issue.",
    "general_maintenance": "Insufficient evidence to identify a specific root cause; technician inspection is required.",
}

ACTIONS = {
    "gearbox_vibration": "Inspect gearbox vibration spectrum, lubrication condition, oil contamination and bearing temperature; follow site safety isolation procedures.",
    "inverter_thermal": "Check inverter alarms, cooling fans/filters, temperature trend and loading; inspect before returning to service.",
    "inspection_compliance": "Assign the pending inspection, verify the approved checklist and record completion evidence.",
    "data_gap": "Check SCADA gateway/network health, data timestamps and source connectivity; backfill data if available.",
    "pitch_system": "Review pitch-system alarms and actuator/sensor status; inspect event logs and perform a controlled functional check.",
    "grid_curtailment": "Verify the curtailment instruction at source, integration mapping and dashboard refresh; reconcile the affected interval.",
    "general_maintenance": "Collect asset ID, alarm codes, timestamps and operating context; request a qualified technician assessment.",
}

ROUTES = {
    "gearbox_vibration": "Wind Turbine Mechanical Maintenance",
    "inverter_thermal": "Solar Electrical / Inverter Team",
    "inspection_compliance": "Hydro Operations & Compliance",
    "data_gap": "SCADA / OT Support Team",
    "pitch_system": "Wind Turbine Controls Team",
    "grid_curtailment": "Grid Operations / SCADA Team",
    "general_maintenance": "Central Maintenance Triage",
}

CRITICAL_TERMS = ["smoke", "fire", "electric shock", "sparking", "person injured", "oil leak", "gearbox failure", "emergency stop"]
HIGH_TERMS = ["repeatedly", "tripping", "abnormal noise", "high temperature", "overheating", "vibration", "pitch error", "shutdown"]
LOW_TERMS = ["cosmetic", "minor paint", "label damaged"]


def _contains(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term in text]


class TriageEngine:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.examples = self._load_data()
        self.dataset_size = len(self.examples)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.training_matrix = self.vectorizer.fit_transform([r["description"] for r in self.examples]) if self.examples else None

    def _load_data(self) -> list[dict[str, str]]:
        with self.data_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["asset_type"] = row["asset_type"].strip().lower()
            row["description"] = re.sub(r"\s+", " ", row["description"]).strip()
            row["target_severity"] = row["target_severity"].strip().lower()
            row["target_category"] = row["target_category"].strip().lower()
        return rows

    def _classify(self, description: str) -> tuple[str, float, str, list[dict[str, Any]]]:
        text = description.lower()
        minor_issue = bool(_contains(text, LOW_TERMS))
        matched = [] if minor_issue else [
            (category, terms) for category, terms in CATEGORY_RULES
            if _contains(text, terms)
        ]

        if minor_issue:
            category = "general_maintenance"
            method = "minor-maintenance rules"
            confidence = 0.85
        elif matched:
            category = matched[0][0]
            method = "keyword rules"
            confidence = min(0.96, 0.70 + 0.08 * len(_contains(text, matched[0][1])))
        elif self.training_matrix is not None:
            query_vec = self.vectorizer.transform([description])
            scores = cosine_similarity(query_vec, self.training_matrix)[0]
            best = int(scores.argmax())
            confidence = float(scores[best])
            category = self.examples[best]["target_category"] if confidence >= 0.12 else "general_maintenance"
            method = "TF-IDF nearest-example similarity" if category != "general_maintenance" else "low-evidence fallback"
        else:
            category, confidence, method = "general_maintenance", 0.0, "fallback"
        evidence = []
        if self.training_matrix is not None:
            scores = cosine_similarity(self.vectorizer.transform([description]), self.training_matrix)[0]
            for idx in scores.argsort()[::-1][:3]:
                evidence.append({"ticket_id": self.examples[int(idx)]["ticket_id"], "category": self.examples[int(idx)]["target_category"], "similarity": round(float(scores[int(idx)]), 3)})
        return category, round(float(confidence), 3), method, evidence

    def _score_severity(self, text: str, category: str) -> tuple[str, list[str]]:
        lowered = text.lower()
        critical = _contains(lowered, CRITICAL_TERMS)
        high = _contains(lowered, HIGH_TERMS)
        low = _contains(lowered, LOW_TERMS)
        reasons: list[str] = []
        if critical:
            reasons.append("Critical safety/equipment-risk phrase detected: " + ", ".join(critical))
            return "critical", reasons
        if category == "gearbox_vibration" and "increased" in lowered and "abnormal noise" in lowered:
            reasons.append("Increased gearbox vibration with abnormal noise matches the supplied critical-risk example.")
            return "critical", reasons
        if high:
            reasons.append("High-risk operating symptom detected: " + ", ".join(high))
            return "high", reasons
        if low:
            reasons.append("Only a minor/cosmetic issue was identified.")
            return "low", reasons
        if category in {"data_gap", "grid_curtailment", "inspection_compliance"}:
            reasons.append("Operational/compliance issue without an explicit immediate safety symptom.")
            return "medium", reasons
        reasons.append("No explicit severity trigger found; conservative default is medium for human review.")
        return "medium", reasons

    def _route(self, severity: str, category: str) -> dict[str, str]:
        team = ROUTES.get(category, ROUTES["general_maintenance"])
        if severity == "critical":
            return {"team": "Site Control Room + Emergency Maintenance Lead", "priority_queue": "P1 - Immediate", "sla_guidance": "Escalate immediately under site emergency procedures"}
        if severity == "high":
            return {"team": team, "priority_queue": "P2 - Urgent", "sla_guidance": "Prompt response; confirm site-specific SLA"}
        if severity == "medium":
            return {"team": team, "priority_queue": "P3 - Planned", "sla_guidance": "Schedule within the maintenance planning window"}
        return {"team": team, "priority_queue": "P4 - Low", "sla_guidance": "Add to routine maintenance backlog"}

    def process(self, description: str, asset_type: str = "unknown", ticket_id: str | None = None) -> dict[str, Any]:
        asset = asset_type.strip().lower() or "unknown"
        if asset not in VALID_ASSETS:
            asset = "unknown"
        category, confidence, method, evidence = self._classify(description)
        severity, severity_reasons = self._score_severity(description, category)
        route = self._route(severity, category)
        # Agent-style orchestration: tools are called in sequence and each result feeds the next step.
        steps = [
            {"step": 1, "tool": "classify_ticket", "output": {"category": category, "method": method}},
            {"step": 2, "tool": "score_severity", "output": {"severity": severity, "reasons": severity_reasons}},
            {"step": 3, "tool": "infer_root_cause", "output": {"hypothesis": ROOT_CAUSES.get(category, ROOT_CAUSES["general_maintenance"])}},
            {"step": 4, "tool": "recommend_action", "output": {"action": ACTIONS.get(category, ACTIONS["general_maintenance"])}},
            {"step": 5, "tool": "route_ticket", "output": route},
        ]
        return {
            "ticket_id": ticket_id or "NEW-TICKET",
            "input": {"asset_type": asset, "description": description.strip()},
            "classification": {"category": category, "confidence": confidence, "method": method},
            "severity": severity,
            "severity_reasons": severity_reasons,
            "probable_root_cause": ROOT_CAUSES.get(category, ROOT_CAUSES["general_maintenance"]),
            "recommended_next_action": ACTIONS.get(category, ACTIONS["general_maintenance"]),
            "routing": route,
            "retrieval_evidence": evidence,
            "agent_trace": steps,
            "human_review_required": True,
            "disclaimer": "Root cause is a hypothesis, not a confirmed diagnosis. Verify on site before taking action.",
        }

    def evaluate(self) -> dict[str, Any]:
        """Leave-one-out TF-IDF category retrieval check on the six supplied rows."""
        if len(self.examples) < 2:
            return {"status": "not_evaluable", "reason": "At least two labelled examples are required."}
        texts = [r["description"] for r in self.examples]
        labels = [r["target_category"] for r in self.examples]
        predictions = []
        for i, text in enumerate(texts):
            train_texts = [t for j, t in enumerate(texts) if j != i]
            train_labels = [lab for j, lab in enumerate(labels) if j != i]
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            matrix = vec.fit_transform(train_texts)
            scores = cosine_similarity(vec.transform([text]), matrix)[0]
            predictions.append(train_labels[int(scores.argmax())])
        correct = sum(p == y for p, y in zip(predictions, labels))
        accuracy = correct / len(labels)
        per_class = {}
        for label in sorted(set(labels)):
            tp = sum(p == label and y == label for p, y in zip(predictions, labels))
            fp = sum(p == label and y != label for p, y in zip(predictions, labels))
            fn = sum(p != label and y == label for p, y in zip(predictions, labels))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "support": labels.count(label)}
        rule_predictions = [self._classify(row["description"])[0] for row in self.examples]
        rule_correct = sum(p == y for p, y in zip(rule_predictions, labels))
        return {
            "status": "completed_with_small_sample_warning",
            "method": "Hybrid keyword rules with TF-IDF nearest-example fallback",
            "sample_count": len(labels),
            "rule_based_example_coverage": round(rule_correct / len(labels), 3),
            "rule_based_example_matches": rule_correct,
            "rule_based_note": "This is an in-sample coverage check because rules were designed from the tiny supplied examples; it is not a held-out accuracy estimate.",
            "tfidf_leave_one_out_correct": correct,
            "tfidf_leave_one_out_accuracy": round(accuracy, 3),
            "tfidf_leave_one_out_macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(per_class), 3),
            "per_class": per_class,
            "predictions": [{"ticket_id": self.examples[i]["ticket_id"], "actual": labels[i], "tfidf_loo_predicted": predictions[i], "tfidf_loo_correct": labels[i] == predictions[i], "hybrid_rule_predicted": rule_predictions[i], "hybrid_rule_match": labels[i] == rule_predictions[i]} for i in range(len(labels))],
            "warning": "Only six examples are supplied and each category appears once. In-sample rule coverage is not generalization evidence, and TF-IDF leave-one-out performance is not meaningful with one example per class. Add a larger, representative, independently labelled dataset before production use.",
        }
