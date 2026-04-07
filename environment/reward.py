"""Multi-signal reward computation for the Clinical Note Scribe environment.

Reward formula (all weights sum to 1.0 before penalties):

  weighted_sum = grader_score       × 0.60   (clinical accuracy from task grader)
               + conciseness_bonus  × 0.10   (1.0 if note ≤ 400 words, else 0.0)
               + safe_language_score× 0.15   (1.0 if no unsafe-certainty phrases)
               + format_valid       × 0.15   (1.0 if SOAP JSON is well-formed)

Deductions (applied after weighted sum):
  - 0.05 × max(0, step_count - 3)   (penalty for excessive clarification steps)
  - 0.10 × len(errors_so_far)        (penalty for each invalid action)

Final value is clamped to [0.0, 1.0].
"""

from __future__ import annotations

import re
from typing import Any, Optional

from environment.models import Action, Reward, SOAPNote


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

W_GRADER      = 0.60
W_CONCISE     = 0.10
W_SAFE_LANG   = 0.15
W_FORMAT      = 0.15

# Deduction constants
STEP_PENALTY_RATE   = 0.05   # per step beyond FREE_STEPS
FREE_STEPS          = 3
ERROR_PENALTY_RATE  = 0.10   # per item in errors_so_far

# Conciseness threshold
WORD_LIMIT = 400

# Phrases that indicate unsafe clinical certainty
# (over-confident language that a scribe should avoid in a note)
_UNSAFE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpatient definitely has\b",
        r"\bdiagnosis is certain\b",
        r"\bno doubt\b",
        r"\babsolutely confirmed\b",
        r"\b100%\s+certain\b",
        r"\bwill definitely\b",
        r"\bguaranteed to\b",
        r"\bcannot be\s+\w+\s+else\b",
        r"\bwithout question\b",
        r"\bthis is clearly\b",
    ]
]


# ---------------------------------------------------------------------------
# Sub-signal helpers
# ---------------------------------------------------------------------------

def _conciseness_bonus(soap_note: Optional[SOAPNote]) -> float:
    """Return 1.0 if the total SOAP note word count is at or below WORD_LIMIT."""
    if soap_note is None:
        return 0.0
    text = " ".join([
        soap_note.subjective,
        soap_note.objective,
        soap_note.assessment,
        soap_note.plan,
    ])
    word_count = len(text.split())
    return 1.0 if word_count <= WORD_LIMIT else 0.0


def _safe_language_score(soap_note: Optional[SOAPNote]) -> float:
    """Return 1.0 if no unsafe-certainty phrases are found in the SOAP note."""
    if soap_note is None:
        return 1.0   # no note submitted → no unsafe language
    text = " ".join([
        soap_note.subjective,
        soap_note.objective,
        soap_note.assessment,
        soap_note.plan,
    ])
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(text):
            return 0.0
    return 1.0


def _format_valid(action: Action) -> float:
    """Return 1.0 if the submitted note has all required non-empty SOAP fields.

    This acts as a lightweight structural / «JSON well-formed» check:
    each of S, O, A, P must be a non-empty string, and the action_type
    must be ``submit_note``.
    """
    if action.action_type != "submit_note":
        return 1.0   # non-submission actions are not graded on format
    if action.soap_note is None:
        return 0.0
    soap = action.soap_note
    fields = [soap.subjective, soap.objective, soap.assessment, soap.plan]
    if all(isinstance(f, str) and f.strip() for f in fields):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    action: Action,
    grader_score: float,
    step_count: int,
    errors_so_far: list[str],
    *,
    done: bool = False,
    info: Optional[dict[str, Any]] = None,
) -> Reward:
    """Compute the multi-signal reward for a completed step.

    Parameters
    ----------
    action:
        The action that was just executed.
    grader_score:
        Clinical-accuracy score returned by the task-specific grader (0.0–1.0).
        Use 0.0 for non-submission actions.
    step_count:
        Total number of steps taken so far in the episode (including this one).
    errors_so_far:
        List of error messages accumulated during the episode.
    done:
        Whether the episode ended with this step.
    info:
        Optional auxiliary metadata dict to include in the Reward.

    Returns
    -------
    Reward
        Fully populated Reward with ``value`` and ``signals`` breakdown.
    """
    grader_score = max(0.0, min(1.0, grader_score))

    # ---- per-signal scores ----
    conciseness  = _conciseness_bonus(action.soap_note)
    safe_lang    = _safe_language_score(action.soap_note)
    fmt          = _format_valid(action)

    # ---- weighted sum ----
    weighted = (
        grader_score * W_GRADER
        + conciseness  * W_CONCISE
        + safe_lang    * W_SAFE_LANG
        + fmt          * W_FORMAT
    )

    # ---- deductions ----
    extra_steps   = max(0, step_count - FREE_STEPS)
    step_penalty  = extra_steps * STEP_PENALTY_RATE
    error_penalty = len(errors_so_far) * ERROR_PENALTY_RATE

    raw = weighted - step_penalty - error_penalty

    # ---- clamp ----
    value = max(0.0, min(1.0, raw))

    signals: dict[str, float] = {
        # positive contributions
        "grader_score":        round(grader_score * W_GRADER,   4),
        "conciseness_bonus":   round(conciseness  * W_CONCISE,  4),
        "safe_language_score": round(safe_lang     * W_SAFE_LANG, 4),
        "format_valid":        round(fmt           * W_FORMAT,   4),
        # deductions (stored as negative numbers for clarity)
        "step_penalty":        round(-step_penalty,  4),
        "error_penalty":       round(-error_penalty, 4),
        # raw sub-signal values (unweighted, for introspection)
        "_grader_score_raw":        round(grader_score, 4),
        "_conciseness_raw":         round(conciseness,  4),
        "_safe_language_raw":       round(safe_lang,    4),
        "_format_valid_raw":        round(fmt,          4),
        "_extra_steps":             float(extra_steps),
        "_error_count":             float(len(errors_so_far)),
    }

    return Reward(
        value=round(value, 4),
        signals=signals,
        done=done,
        info=info or {},
    )
