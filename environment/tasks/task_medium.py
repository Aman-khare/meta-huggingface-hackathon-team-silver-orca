"""Medium task — chronic disease follow-up.

Grader is intentionally left unimplemented.
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
# Grader (not yet implemented)
# ---------------------------------------------------------------------------

def grade_medium(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    """Score a submitted SOAP note against the medium-task rubric.

    Parameters
    ----------
    soap_note:
        The agent's submitted clinical note.
    task:
        The task definition dict (``MEDIUM_TASK``).

    Returns
    -------
    dict mapping signal names → float scores in [0, 1].

    Raises
    ------
    NotImplementedError
        Grader has not been implemented yet.
    """
    raise NotImplementedError("Medium-task grader is not yet implemented.")
