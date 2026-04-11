"""Multi-signal reward computation for the Clinical Note Scribe environment."""

from __future__ import annotations

import re
from typing import Any, Optional

from environment.models import Action, Reward, SOAPNote

# Reward weights (sum to 1.0)
W_GRADER, W_CONCISE, W_SAFE_LANG, W_FORMAT = 0.60, 0.10, 0.15, 0.15

# Deductions
STEP_PENALTY_RATE = 0.05
FREE_STEPS = 3
ERROR_PENALTY_RATE = 0.10
WORD_LIMIT = 400

# Pre-compiled unsafe clinical certainty patterns
_UNSAFE_RE = re.compile(
    r"\bpatient definitely has\b"
    r"|\bdiagnosis is certain\b"
    r"|\bno doubt\b"
    r"|\babsolutely confirmed\b"
    r"|\b100%\s+certain\b"
    r"|\bwill definitely\b"
    r"|\bguaranteed to\b"
    r"|\bcannot be\s+\w+\s+else\b"
    r"|\bwithout question\b"
    r"|\bthis is clearly\b",
    re.IGNORECASE,
)


def _soap_text(soap: Optional[SOAPNote]) -> Optional[str]:
    """Join all SOAP fields into one string. Returns None if no note."""
    if soap is None:
        return None
    return f"{soap.subjective} {soap.objective} {soap.assessment} {soap.plan}"


def compute_reward(
    action: Action,
    grader_score: float,
    step_count: int,
    errors_so_far: list[str],
    *,
    done: bool = False,
    info: Optional[dict[str, Any]] = None,
) -> Reward:
    """Compute the multi-signal reward for a completed step."""
    grader_score = max(0.0, min(1.0, grader_score))
    text = _soap_text(action.soap_note)

    # Sub-signals
    conciseness = 1.0 if text and len(text.split()) <= WORD_LIMIT else (0.0 if text else 0.0)
    safe_lang = 0.0 if (text and _UNSAFE_RE.search(text)) else 1.0
    fmt = (
        1.0
        if action.action_type != "submit_note"
        else (
            1.0
            if action.soap_note and all(
                getattr(action.soap_note, f).strip()
                for f in ("subjective", "objective", "assessment", "plan")
            )
            else 0.0
        )
    )

    # Weighted sum minus deductions, clamped to (0, 1)
    extra_steps = max(0, step_count - FREE_STEPS)
    raw = (
        grader_score * W_GRADER
        + conciseness * W_CONCISE
        + safe_lang * W_SAFE_LANG
        + fmt * W_FORMAT
        - extra_steps * STEP_PENALTY_RATE
        - len(errors_so_far) * ERROR_PENALTY_RATE
    )
    value = round(max(0.01, min(0.99, raw)), 4)

    return Reward(
        value=value,
        signals={
            "grader_score": round(grader_score * W_GRADER, 4),
            "conciseness_bonus": round(conciseness * W_CONCISE, 4),
            "safe_language_score": round(safe_lang * W_SAFE_LANG, 4),
            "format_valid": round(fmt * W_FORMAT, 4),
            "step_penalty": round(-extra_steps * STEP_PENALTY_RATE, 4),
            "error_penalty": round(-len(errors_so_far) * ERROR_PENALTY_RATE, 4),
            "_grader_score_raw": round(grader_score, 4),
            "_conciseness_raw": round(conciseness, 4),
            "_safe_language_raw": round(safe_lang, 4),
            "_format_valid_raw": round(fmt, 4),
            "_extra_steps": float(extra_steps),
            "_error_count": float(len(errors_so_far)),
        },
        done=done,
        info=info or {},
    )
