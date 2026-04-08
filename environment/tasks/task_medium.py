"""Medium task — chronic disease follow-up.

Grader uses keyword-based clinical rubric scoring to evaluate the SOAP note
against expected findings from a Type 2 Diabetes / Hypertension follow-up.
"""

from __future__ import annotations

from typing import Any

from environment.models import SOAPNote


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

MEDIUM_TASK: dict[str, Any] = {
    "task_id": "medium_chronic_disease_followup",
    "description": "Generate a SOAP note for a Type 2 Diabetes follow-up visit.",
    "transcript_file": "data/transcripts/medium.txt",
    "patient_context": {
        "patient_id": "P-2045",
        "name": "Robert Smith",
        "age": 58,
        "sex": "M",
        "known_conditions": ["Type 2 Diabetes Mellitus", "Hypertension"],
        "current_medications": [
            "Metformin 1000 mg BID",
            "Lisinopril 20 mg daily",
            "Atorvastatin 40 mg daily",
        ],
        "allergies": [],
        "recent_labs": {
            "HbA1c": "7.8%",
            "fasting_glucose": "156 mg/dL",
            "creatinine": "1.1 mg/dL",
            "eGFR": "78 mL/min",
            "LDL": "102 mg/dL",
        },
    },
    "max_steps": 8,
}


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade_medium(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    """Score a submitted SOAP note against the medium-task rubric.

    Checks for mention of dietary habits, HbA1c lab values, core diagnoses,
    and medication adjustments (glipizide, lisinopril uptitration).

    Returns
    -------
    dict mapping signal names to float scores in [0, 1].
    """
    text_s = soap_note.subjective.lower()
    text_o = soap_note.objective.lower()
    text_a = soap_note.assessment.lower()
    text_p = soap_note.plan.lower()

    # 1. Subjective — dietary habits / statin gap
    s_score = 0.0
    if "restaurant" in text_s or "diet" in text_s or "eating" in text_s:
        s_score += 0.5
    if "statin" in text_s or "gap" in text_s or "missed" in text_s:
        s_score += 0.5

    # 2. Objective — HbA1c values
    o_score = 0.0
    if "7.8" in text_o or "7.2" in text_o or "a1c" in text_o or "hba1c" in text_o:
        o_score += 0.5
    if "156" in text_o or "fasting glucose" in text_o or "glucose" in text_o:
        o_score += 0.5

    # 3. Assessment — core diagnoses
    a_score = 0.0
    if "diabetes" in text_a or "t2dm" in text_a or "dm" in text_a:
        a_score += 0.5
    if "hypertension" in text_a or "htn" in text_a or "blood pressure" in text_a:
        a_score += 0.5

    # 4. Plan — medication changes
    p_score = 0.0
    if "glipizide" in text_p and ("5" in text_p or "add" in text_p):
        p_score += 0.5
    if "lisinopril" in text_p and ("40" in text_p or "increase" in text_p or "uptitrat" in text_p):
        p_score += 0.5

    return {
        "subjective_accuracy": max(0.01, min(s_score, 0.99)),
        "objective_accuracy": max(0.01, min(o_score, 0.99)),
        "assessment_accuracy": max(0.01, min(a_score, 0.99)),
        "plan_accuracy": max(0.01, min(p_score, 0.99)),
    }
