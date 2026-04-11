"""Medium task — chronic disease follow-up."""

from __future__ import annotations
from typing import Any
from environment.models import SOAPNote

MEDIUM_TASK: dict[str, Any] = {
    "task_id": "medium_chronic_disease_followup",
    "description": "Generate a SOAP note for a Type 2 Diabetes follow-up visit.",
    "transcript_file": "data/transcripts/medium.txt",
    "patient_context": {
        "patient_id": "P-2045", "name": "Robert Smith", "age": 58, "sex": "M",
        "known_conditions": ["Type 2 Diabetes Mellitus", "Hypertension"],
        "current_medications": ["Metformin 1000 mg BID", "Lisinopril 20 mg daily", "Atorvastatin 40 mg daily"],
        "allergies": [],
        "recent_labs": {"HbA1c": "7.8%", "fasting_glucose": "156 mg/dL", "creatinine": "1.1 mg/dL", "eGFR": "78 mL/min", "LDL": "102 mg/dL"},
    },
    "max_steps": 8,
}


def grade_medium(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    s, o, a, p = (soap_note.subjective.lower(), soap_note.objective.lower(),
                  soap_note.assessment.lower(), soap_note.plan.lower())

    s_score = 0.5 * any(k in s for k in ("restaurant", "diet", "eating")) \
            + 0.5 * any(k in s for k in ("statin", "gap", "missed"))
    o_score = 0.5 * any(k in o for k in ("7.8", "7.2", "a1c", "hba1c")) \
            + 0.5 * any(k in o for k in ("156", "fasting glucose", "glucose"))
    a_score = 0.5 * any(k in a for k in ("diabetes", "t2dm", "dm")) \
            + 0.5 * any(k in a for k in ("hypertension", "htn", "blood pressure"))
    p_score = 0.5 * ("glipizide" in p and any(k in p for k in ("5", "add"))) \
            + 0.5 * ("lisinopril" in p and any(k in p for k in ("40", "increase", "uptitrat")))

    clamp = lambda v: max(0.01, min(v, 0.99))
    return {
        "subjective_accuracy": clamp(s_score),
        "objective_accuracy": clamp(o_score),
        "assessment_accuracy": clamp(a_score),
        "plan_accuracy": clamp(p_score),
    }
