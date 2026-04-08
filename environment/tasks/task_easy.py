"""Easy task — routine check-up.

Grader uses keyword-based clinical rubric scoring to evaluate the SOAP note
against expected findings from a simple cold / blood pressure check visit.
"""

from __future__ import annotations

from typing import Any

from environment.models import SOAPNote


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

EASY_TASK: dict[str, Any] = {
    "task_id": "easy_routine_checkup",
    "description": "Generate a SOAP note for a routine annual check-up visit.",
    "transcript_file": "data/transcripts/easy.txt",
    "patient_context": {
        "patient_id": "P-1001",
        "name": "Jane Doe",
        "age": 34,
        "sex": "F",
        "known_conditions": [],
        "current_medications": [],
        "allergies": ["Penicillin"],
    },
    "max_steps": 5,
}


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade_easy(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    """Score a submitted SOAP note against the easy-task rubric.

    Checks for mention of key clinical findings from the transcript:
    chief complaints, vitals, viral URI assessment, and supportive plan.

    Returns
    -------
    dict mapping signal names to float scores in [0, 1].
    """
    text_s = soap_note.subjective.lower()
    text_o = soap_note.objective.lower()
    text_a = soap_note.assessment.lower()
    text_p = soap_note.plan.lower()

    # 1. Subjective — chief complaints
    s_score = 0.0
    if "sore throat" in text_s or "runny nose" in text_s or "congestion" in text_s:
        s_score += 0.5
    if "5 days" in text_s or "five days" in text_s or "headache" in text_s:
        s_score += 0.5

    # 2. Objective — vitals
    o_score = 0.0
    if "118/76" in text_o or "118 over 76" in text_o or "blood pressure" in text_o:
        o_score += 0.5
    if "72" in text_o or "heart rate" in text_o or "lungs clear" in text_o:
        o_score += 0.5

    # 3. Assessment — viral URI
    a_score = 0.0
    if "viral" in text_a or "uri" in text_a or "upper respiratory" in text_a:
        a_score += 1.0

    # 4. Plan — supportive care
    p_score = 0.0
    if "fluids" in text_p or "rest" in text_p or "hydrat" in text_p:
        p_score += 0.5
    if "dayquil" in text_p or "follow" in text_p or "return" in text_p:
        p_score += 0.5

    return {
        "subjective_accuracy": min(s_score, 1.0),
        "objective_accuracy": min(o_score, 1.0),
        "assessment_accuracy": min(a_score, 1.0),
        "plan_accuracy": min(p_score, 1.0),
    }
