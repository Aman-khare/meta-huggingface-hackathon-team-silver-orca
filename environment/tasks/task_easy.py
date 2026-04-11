"""Easy task — routine check-up."""

from __future__ import annotations
from typing import Any
from environment.models import SOAPNote

EASY_TASK: dict[str, Any] = {
    "task_id": "easy_routine_checkup",
    "description": "Generate a SOAP note for a routine annual check-up visit.",
    "transcript_file": "data/transcripts/easy.txt",
    "patient_context": {
        "patient_id": "P-1001", "name": "Jane Doe", "age": 34, "sex": "F",
        "known_conditions": [], "current_medications": [], "allergies": ["Penicillin"],
    },
    "max_steps": 5,
}


def grade_easy(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    s, o, a, p = (soap_note.subjective.lower(), soap_note.objective.lower(),
                  soap_note.assessment.lower(), soap_note.plan.lower())

    s_score = 0.5 * any(k in s for k in ("sore throat", "runny nose", "congestion")) \
            + 0.5 * any(k in s for k in ("5 days", "five days", "headache"))
    o_score = 0.5 * any(k in o for k in ("118/76", "118 over 76", "blood pressure")) \
            + 0.5 * any(k in o for k in ("72", "heart rate", "lungs clear"))
    a_score = 1.0 * any(k in a for k in ("viral", "uri", "upper respiratory"))
    p_score = 0.5 * any(k in p for k in ("fluids", "rest", "hydrat")) \
            + 0.5 * any(k in p for k in ("dayquil", "follow", "return"))

    clamp = lambda v: max(0.01, min(v, 0.99))
    return {
        "subjective_accuracy": clamp(s_score),
        "objective_accuracy": clamp(o_score),
        "assessment_accuracy": clamp(a_score),
        "plan_accuracy": clamp(p_score),
    }
