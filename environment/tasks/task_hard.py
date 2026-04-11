"""Hard task — complex ER visit."""

from __future__ import annotations
from typing import Any
from environment.models import SOAPNote

HARD_TASK: dict[str, Any] = {
    "task_id": "hard_complex_er_visit",
    "description": "Generate a SOAP note for a complex ER visit with chest pain, PE differential, and contrast allergy.",
    "transcript_file": "data/transcripts/hard.txt",
    "patient_context": {
        "patient_id": "P-3782", "name": "Maria Garcia", "age": 72, "sex": "F",
        "known_conditions": ["Coronary Artery Disease", "Atrial Fibrillation", "Chronic Kidney Disease Stage 3", "Osteoarthritis"],
        "current_medications": ["Aspirin 81 mg daily", "Warfarin 5 mg daily", "Metoprolol 50 mg BID", "Furosemide 40 mg daily", "Amlodipine 5 mg daily"],
        "allergies": ["Sulfa drugs", "Contrast dye"],
        "recent_labs": {"troponin_I": "0.08 ng/mL", "BNP": "450 pg/mL", "creatinine": "1.9 mg/dL", "eGFR": "34 mL/min", "INR": "2.6", "hemoglobin": "10.2 g/dL"},
        "vitals_on_arrival": {"BP": "168/94 mmHg", "HR": "112 bpm (irregular)", "RR": "22 breaths/min", "SpO2": "91% on room air", "Temp": "37.2°C"},
    },
    "max_steps": 10,
}


def grade_hard(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    s, o, a, p = (soap_note.subjective.lower(), soap_note.objective.lower(),
                  soap_note.assessment.lower(), soap_note.plan.lower())

    s_score = 0.5 * any(k in s for k in ("chest pain", "shortness of breath", "sob")) \
            + 0.5 * any(k in s for k in ("nitroglycerin", "contradict", "denied"))
    o_score = 0.5 * any(k in o for k in ("d-dimer", "1840", "d dimer")) \
            + 0.5 * any(k in o for k in ("allergy", "contrast", "troponin"))
    a_score = 0.5 * any(k in a for k in ("acs", "acute coronary", "coronary", "ischemia")) \
            + 0.5 * any(k in a for k in ("pe", "pulmonary embolism", "embolism"))
    p_score = 0.5 * any(k in p for k in ("v/q", "ventilation", "perfusion")) \
            + 0.5 * any(k in p for k in ("icu", "admit", "cardiac"))

    clamp = lambda v: max(0.01, min(v, 0.99))
    return {
        "subjective_accuracy": clamp(s_score),
        "objective_accuracy": clamp(o_score),
        "assessment_accuracy": clamp(a_score),
        "plan_accuracy": clamp(p_score),
    }
