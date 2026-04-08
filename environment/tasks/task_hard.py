"""Hard task — complex ER visit.

Grader uses keyword-based clinical rubric scoring to evaluate the SOAP note
against expected findings from a complex ER visit with overlapping chest pain,
SOB, and a possible PE complicated by a contrast dye allergy.
"""

from __future__ import annotations

from typing import Any

from environment.models import SOAPNote


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

HARD_TASK: dict[str, Any] = {
    "task_id": "hard_complex_er_visit",
    "description": (
        "Generate a SOAP note for a complex emergency-room visit involving "
        "chest pain, polytrauma assessment, and multiple co-morbidities."
    ),
    "transcript_file": "data/transcripts/hard.txt",
    "patient_context": {
        "patient_id": "P-3782",
        "name": "Maria Garcia",
        "age": 72,
        "sex": "F",
        "known_conditions": [
            "Coronary Artery Disease",
            "Atrial Fibrillation",
            "Chronic Kidney Disease Stage 3",
            "Osteoarthritis",
        ],
        "current_medications": [
            "Aspirin 81 mg daily",
            "Warfarin 5 mg daily",
            "Metoprolol 50 mg BID",
            "Furosemide 40 mg daily",
            "Amlodipine 5 mg daily",
        ],
        "allergies": ["Sulfa drugs", "Contrast dye"],
        "recent_labs": {
            "troponin_I": "0.08 ng/mL",
            "BNP": "450 pg/mL",
            "creatinine": "1.9 mg/dL",
            "eGFR": "34 mL/min",
            "INR": "2.6",
            "hemoglobin": "10.2 g/dL",
        },
        "vitals_on_arrival": {
            "BP": "168/94 mmHg",
            "HR": "112 bpm (irregular)",
            "RR": "22 breaths/min",
            "SpO2": "91% on room air",
            "Temp": "37.2°C",
        },
    },
    "max_steps": 10,
}


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade_hard(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    """Score a submitted SOAP note against the hard-task rubric.

    Checks for chest pain / SOB and the nitroglycerin contradiction (subjective),
    D-dimer and contrast allergy (objective), ACS vs PE differential (assessment),
    and V/Q scan + ICU admission (plan).

    Returns
    -------
    dict mapping signal names to float scores in [0, 1].
    """
    text_s = soap_note.subjective.lower()
    text_o = soap_note.objective.lower()
    text_a = soap_note.assessment.lower()
    text_p = soap_note.plan.lower()

    # 1. Subjective — catching the contradiction and presenting complaints
    s_score = 0.0
    if "chest pain" in text_s or "shortness of breath" in text_s or "sob" in text_s:
        s_score += 0.5
    if "nitroglycerin" in text_s or "contradict" in text_s or "denied" in text_s:
        s_score += 0.5

    # 2. Objective — elevated D-dimer and allergy awareness
    o_score = 0.0
    if "d-dimer" in text_o or "1840" in text_o or "d dimer" in text_o:
        o_score += 0.5
    if "allergy" in text_o or "contrast" in text_o or "troponin" in text_o:
        o_score += 0.5

    # 3. Assessment — the dual differential (ACS vs PE)
    a_score = 0.0
    if "acs" in text_a or "acute coronary" in text_a or "coronary" in text_a or "ischemia" in text_a:
        a_score += 0.5
    if "pe" in text_a or "pulmonary embolism" in text_a or "embolism" in text_a:
        a_score += 0.5

    # 4. Plan — adapting to the allergy (V/Q scan) and admission
    p_score = 0.0
    if "v/q" in text_p or "ventilation" in text_p or "perfusion" in text_p:
        p_score += 0.5
    if "icu" in text_p or "admit" in text_p or "cardiac" in text_p:
        p_score += 0.5

    return {
        "subjective_accuracy": min(s_score, 1.0),
        "objective_accuracy": min(o_score, 1.0),
        "assessment_accuracy": min(a_score, 1.0),
        "plan_accuracy": min(p_score, 1.0),
    }
