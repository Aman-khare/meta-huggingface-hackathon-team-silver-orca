"""ClinicalNoteScribeEnv — core environment loop.

Implements the ``reset() → Observation``, ``step(Action) → (Observation, Reward, bool, dict)``,
and ``state() → EnvironmentState`` interface required by the OpenEnv spec.

Structured logging
------------------
Every episode emits exactly three kinds of JSON log lines to **stdout**:

- ``{"event": "START", "task_id": "...", "timestamp": ...}``
- ``{"event": "STEP",  "step": N, "action_type": "...", "reward": R}``
- ``{"event": "END",   "task_id": "...", "final_score": S}``

The OpenEnv validator scrapes ``[START]``, ``[STEP]``, ``[END]`` keywords.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from environment.models import Action, EnvironmentState, Observation, Reward, SOAPNote
from environment.reward import compute_reward
from environment.tasks import GRADER_REGISTRY, TASK_REGISTRY

logger = logging.getLogger("clinical_note_scribe")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_transcript(transcript_path: str) -> str:
    """Load a transcript text file from *project-root-relative* path."""
    base = Path(__file__).resolve().parent.parent  # clinical-note-scribe/
    full_path = base / transcript_path
    if full_path.exists():
        return full_path.read_text(encoding="utf-8")
    return f"[Transcript file not found: {transcript_path}]"


def _log_event(event: str, **kwargs: Any) -> None:
    """Emit a structured JSON log line to stdout via the logger."""
    payload: dict[str, Any] = {"event": event, "timestamp": time.time()}
    payload.update(kwargs)
    logger.info(json.dumps(payload))


def _soap_to_text(soap: SOAPNote) -> str:
    """Flatten a SOAPNote into a readable multi-line string."""
    return (
        f"S: {soap.subjective}\n"
        f"O: {soap.objective}\n"
        f"A: {soap.assessment}\n"
        f"P: {soap.plan}"
    )


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class ClinicalNoteScribeEnv:
    """Open-environment wrapper for the clinical note-scribe tasks.

    Lifecycle
    ---------
    1. ``env.reset(task_id)``  → returns initial ``Observation``
    2. ``env.step(action)``    → returns ``(Observation, Reward, done, info)``
    3. ``env.state()``         → returns full ``EnvironmentState`` snapshot

    Parameters
    ----------
    clarify_answers_path:
        Project-root-relative path to the clarification lookup JSON.
    """

    def __init__(
        self,
        clarify_answers_path: str = "data/clarify_answers.json",
    ) -> None:
        self._clarify_answers: dict[str, str] = {}
        base = Path(__file__).resolve().parent.parent
        ca_path = base / clarify_answers_path
        if ca_path.exists():
            self._clarify_answers = json.loads(ca_path.read_text(encoding="utf-8"))

        # Episode state (initialised properly in reset())
        self._task: dict[str, Any] = {}
        self._task_id: str = ""
        self._transcript: str = ""
        self._patient_context: dict[str, Any] = {}
        self._max_steps: int = 10
        self._step_count: int = 0
        self._done: bool = True
        self._current_draft: str | None = None
        self._errors_so_far: list[str] = []
        self._last_reward: Reward | None = None
        self._last_observation: Observation | None = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def reset(self, task_id: str | None = None) -> Observation:
        """Start (or restart) an episode for the given *task_id*.

        Parameters
        ----------
        task_id:
            One of the keys in ``TASK_REGISTRY``.  When ``None`` the first
            registered task is used.

        Returns
        -------
        Observation
            The initial observation for the episode.

        Raises
        ------
        ValueError
            If *task_id* is not found in the registry.
        """
        if task_id is None:
            task_id = next(iter(TASK_REGISTRY))

        if task_id not in TASK_REGISTRY:
            available = ", ".join(TASK_REGISTRY.keys())
            raise ValueError(
                f"Unknown task_id '{task_id}'. Available: {available}"
            )

        self._task = TASK_REGISTRY[task_id]
        self._task_id = task_id
        self._transcript = _load_transcript(self._task["transcript_file"])
        self._patient_context = self._task.get("patient_context", {})
        self._max_steps = self._task.get("max_steps", 10)
        self._step_count = 0
        self._done = False
        self._current_draft = None
        self._errors_so_far = []
        self._last_reward = None

        _log_event("START", task_id=self._task_id)

        obs = self._build_observation()
        self._last_observation = obs
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Execute one agent action and return the resulting observation, reward,
        done flag, and info dict.

        Parameters
        ----------
        action:
            The agent's chosen action.

        Returns
        -------
        tuple[Observation, Reward, bool, dict]
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping again."
            )

        self._step_count += 1
        info: dict[str, Any] = {}

        # ---- dispatch by action type ----
        if action.action_type == "submit_note":
            reward = self._handle_submit(action, info)
        elif action.action_type == "request_clarify":
            reward = self._handle_clarify(action, info)
        elif action.action_type == "revise_section":
            reward = self._handle_revise(action, info)
        else:
            # Should never happen thanks to the Literal type, but be safe
            self._errors_so_far.append(
                f"Unknown action_type: {action.action_type}"
            )
            reward = compute_reward(
                action,
                grader_score=0.0,
                step_count=self._step_count,
                errors_so_far=self._errors_so_far,
                done=False,
                info={"error": "bad_action"},
            )

        # ---- enforce max-step termination ----
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            reward = Reward(
                value=reward.value,
                signals=reward.signals,
                done=True,
                info={**reward.info, "termination_reason": "max_steps_reached"},
            )

        self._last_reward = reward

        _log_event(
            "STEP",
            step=self._step_count,
            action_type=action.action_type,
            reward=reward.value,
        )

        if self._done:
            _log_event(
                "END",
                task_id=self._task_id,
                final_score=reward.value,
            )

        obs = self._build_observation()
        self._last_observation = obs
        return obs, reward, self._done, info

    def state(self) -> EnvironmentState:
        """Return the full internal state snapshot."""
        return EnvironmentState(
            task_id=self._task_id,
            step_count=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            current_draft=self._current_draft,
            errors_so_far=list(self._errors_so_far),
            last_reward=self._last_reward,
            observation=self._last_observation,
        )

    # --------------------------------------------------------------------- #
    # Action handlers
    # --------------------------------------------------------------------- #

    def _handle_submit(self, action: Action, info: dict) -> Reward:
        """Process a ``submit_note`` action."""
        if action.soap_note is None:
            error = "submit_note requires a non-null soap_note."
            self._errors_so_far.append(error)
            return compute_reward(
                action,
                grader_score=0.0,
                step_count=self._step_count,
                errors_so_far=self._errors_so_far,
                done=False,
                info={"error": error},
            )

        self._current_draft = _soap_to_text(action.soap_note)
        self._done = True

        # Attempt to grade via the task-specific grader
        grader = GRADER_REGISTRY.get(self._task_id)
        if grader is None:
            info["warning"] = "No grader registered; returning default reward."
            return compute_reward(
                action,
                grader_score=0.5,
                step_count=self._step_count,
                errors_so_far=self._errors_so_far,
                done=True,
                info=info,
            )

        try:
            raw_signals = grader(action.soap_note, self._task)
            # Grader returns a signals dict; extract a single scalar score
            # as the mean of its values for use as grader_score.
            grader_score = (
                sum(raw_signals.values()) / len(raw_signals)
                if raw_signals else 0.0
            )
            info["grader_signals"] = raw_signals
        except NotImplementedError:
            info["warning"] = "Grader not yet implemented; returning placeholder."
            grader_score = 0.5

        return compute_reward(
            action,
            grader_score=grader_score,
            step_count=self._step_count,
            errors_so_far=self._errors_so_far,
            done=True,
            info=info,
        )

    def _handle_clarify(self, action: Action, info: dict) -> Reward:
        """Process a ``request_clarify`` action."""
        question = (action.clarify_question or "").strip()
        if not question:
            error = "request_clarify requires a non-empty clarify_question."
            self._errors_so_far.append(error)
            return compute_reward(
                action,
                grader_score=0.0,
                step_count=self._step_count,
                errors_so_far=self._errors_so_far,
                done=False,
                info={"error": error},
            )

        # Lookup a canned answer (case-insensitive key match)
        answer = self._clarify_answers.get(question.lower())
        if answer:
            info["clarify_answer"] = answer
        else:
            info["clarify_answer"] = (
                "No additional information available for that question."
            )

        # Clarification steps earn no grader_score; step_penalty accrues naturally
        return compute_reward(
            action,
            grader_score=0.0,
            step_count=self._step_count,
            errors_so_far=self._errors_so_far,
            done=False,
            info=info,
        )

    def _handle_revise(self, action: Action, info: dict) -> Reward:
        """Process a ``revise_section`` action."""
        if action.section is None or action.revision_text is None:
            error = "revise_section requires both 'section' and 'revision_text'."
            self._errors_so_far.append(error)
            return compute_reward(
                action,
                grader_score=0.0,
                step_count=self._step_count,
                errors_so_far=self._errors_so_far,
                done=False,
                info={"error": error},
            )

        # If there is an existing draft, patch the requested section
        if self._current_draft:
            lines = self._current_draft.split("\n")
            prefix = f"{action.section}: "
            patched = False
            for i, line in enumerate(lines):
                if line.startswith(prefix):
                    lines[i] = f"{prefix}{action.revision_text}"
                    patched = True
                    break
            if patched:
                self._current_draft = "\n".join(lines)
            else:
                self._current_draft += f"\n{prefix}{action.revision_text}"
        else:
            self._current_draft = f"{action.section}: {action.revision_text}"

        info["revised_section"] = action.section

        # Revision steps earn no grader_score; deductions still apply
        return compute_reward(
            action,
            grader_score=0.0,
            step_count=self._step_count,
            errors_so_far=self._errors_so_far,
            done=False,
            info=info,
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_observation(self) -> Observation:
        return Observation(
            transcript=self._transcript,
            task_id=self._task_id,
            patient_context=self._patient_context,
            current_draft=self._current_draft,
            errors_so_far=list(self._errors_so_far),
            step_count=self._step_count,
        )
