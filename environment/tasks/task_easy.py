"""Easy task — routine check-up.

Grader is intentionally left unimplemented.
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
# Grader (not yet implemented)
# ---------------------------------------------------------------------------

def grade_easy(soap_note: SOAPNote, task: dict[str, Any]) -> dict[str, float]:
    """Score a submitted SOAP note against the easy-task rubric.

    Parameters
    ----------
    soap_note:
        The agent's submitted clinical note.
    task:
        The task definition dict (``EASY_TASK``).

    Returns
    -------
    dict mapping signal names → float scores in [0, 1].

    Raises
    ------
    NotImplementedError
        Grader has not been implemented yet.
    """
    raise NotImplementedError("Easy-task grader is not yet implemented.")
