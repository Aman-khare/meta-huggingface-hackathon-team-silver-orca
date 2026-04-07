"""Hard task — complex ER visit.

Grader is intentionally left unimplemented.
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
# Grader (not yet implemented)
# ---------------------------------------------------------------------------

def grade_hard(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    """Score a submitted SOAP note against the hard-task rubric.

    Parameters
    ----------
    soap_note:
        The agent's submitted clinical note.
    task:
        The task definition dict (``HARD_TASK``).

    Returns
    -------
    dict mapping signal names → float scores in [0, 1].

    Raises
    ------
    NotImplementedError
        Grader has not been implemented yet.
    """
    raise NotImplementedError("Hard-task grader is not yet implemented.")
