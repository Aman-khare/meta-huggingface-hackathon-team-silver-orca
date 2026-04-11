"""ClinicalNoteScribeEnv — core environment implementing reset/step/state."""

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
_ROOT = Path(__file__).resolve().parent.parent


def _log(event: str, **kw: Any) -> None:
    logger.info(json.dumps({"event": event, "timestamp": time.time(), **kw}))


class ClinicalNoteScribeEnv:
    """OpenEnv-compliant environment for clinical SOAP-note generation."""

    def __init__(self, clarify_answers_path: str = "data/clarify_answers.json") -> None:
        ca = _ROOT / clarify_answers_path
        self._clarify_answers: dict[str, str] = json.loads(ca.read_text()) if ca.exists() else {}
        self._reset_state()

    def _reset_state(self) -> None:
        self._task: dict[str, Any] = {}
        self._task_id = ""
        self._transcript = ""
        self._patient_context: dict[str, Any] = {}
        self._max_steps = 10
        self._step_count = 0
        self._done = True
        self._current_draft: str | None = None
        self._errors_so_far: list[str] = []
        self._last_reward: Reward | None = None
        self._last_observation: Observation | None = None

    def _obs(self) -> Observation:
        return Observation(
            transcript=self._transcript,
            task_id=self._task_id,
            patient_context=self._patient_context,
            current_draft=self._current_draft,
            errors_so_far=list(self._errors_so_far),
            step_count=self._step_count,
        )

    # Public API

    def reset(self, task_id: str | None = None) -> Observation:
        task_id = task_id or next(iter(TASK_REGISTRY))
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {', '.join(TASK_REGISTRY)}")

        self._task = TASK_REGISTRY[task_id]
        self._task_id = task_id
        path = _ROOT / self._task["transcript_file"]
        self._transcript = path.read_text(encoding="utf-8") if path.exists() else f"[Not found: {path}]"
        self._patient_context = self._task.get("patient_context", {})
        self._max_steps = self._task.get("max_steps", 10)
        self._step_count = 0
        self._done = False
        self._current_draft = None
        self._errors_so_far = []
        self._last_reward = None

        _log("START", task_id=task_id)
        self._last_observation = self._obs()
        return self._last_observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        self._step_count += 1
        info: dict[str, Any] = {}

        # Dispatch
        handler = {
            "submit_note": self._submit,
            "request_clarify": self._clarify,
            "revise_section": self._revise,
        }.get(action.action_type)

        if handler:
            reward = handler(action, info)
        else:
            self._errors_so_far.append(f"Unknown action_type: {action.action_type}")
            reward = compute_reward(action, 0.0, self._step_count, self._errors_so_far, done=False, info={"error": "bad_action"})

        # Max-step termination
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            reward = Reward(value=reward.value, signals=reward.signals, done=True,
                            info={**reward.info, "termination_reason": "max_steps_reached"})

        self._last_reward = reward
        _log("STEP", step=self._step_count, action_type=action.action_type, reward=reward.value)
        if self._done:
            _log("END", task_id=self._task_id, final_score=reward.value)

        self._last_observation = self._obs()
        return self._last_observation, reward, self._done, info

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self._task_id, step_count=self._step_count, max_steps=self._max_steps,
            done=self._done, current_draft=self._current_draft,
            errors_so_far=list(self._errors_so_far),
            last_reward=self._last_reward, observation=self._last_observation,
        )

    # Action handlers

    def _submit(self, action: Action, info: dict) -> Reward:
        soap = action.soap_note

        # Fall back to draft
        if soap is None and self._current_draft:
            secs = {}
            for line in self._current_draft.split("\n"):
                for p in ("S: ", "O: ", "A: ", "P: "):
                    if line.startswith(p):
                        secs[p[0]] = line[len(p):]
            if all(k in secs for k in "SOAP"):
                soap = SOAPNote(subjective=secs["S"], objective=secs["O"], assessment=secs["A"], plan=secs["P"])

        if soap is None:
            err = "submit_note requires a non-null soap_note (or a complete draft from revise_section)."
            self._errors_so_far.append(err)
            return compute_reward(action, 0.0, self._step_count, self._errors_so_far, done=False, info={"error": err})

        self._current_draft = f"S: {soap.subjective}\nO: {soap.objective}\nA: {soap.assessment}\nP: {soap.plan}"
        self._done = True

        grader = GRADER_REGISTRY.get(self._task_id)
        if not grader:
            info["warning"] = "No grader registered; returning default reward."
            return compute_reward(action, 0.5, self._step_count, self._errors_so_far, done=True, info=info)

        try:
            signals = grader(soap, self._task)
            score = sum(signals.values()) / len(signals) if signals else 0.0
            info["grader_signals"] = signals
        except Exception as exc:
            info["warning"] = f"Grader error: {exc}"
            score = 0.0

        return compute_reward(action, score, self._step_count, self._errors_so_far, done=True, info=info)

    def _clarify(self, action: Action, info: dict) -> Reward:
        q = (action.clarify_question or "").strip()
        if not q:
            err = "request_clarify requires a non-empty clarify_question."
            self._errors_so_far.append(err)
            return Reward(value=0.0, signals={"error": 1.0}, done=False, info={"error": err})

        info["clarify_answer"] = self._clarify_answers.get(q.lower(), "No additional information available for that question.")
        return Reward(value=0.0, signals={"intermediate_step": 1.0}, done=False, info=info)

    def _revise(self, action: Action, info: dict) -> Reward:
        if action.section is None or action.revision_text is None:
            err = "revise_section requires both 'section' and 'revision_text'."
            self._errors_so_far.append(err)
            return Reward(value=0.0, signals={"error": 1.0}, done=False, info={"error": err})

        prefix = f"{action.section}: "
        if self._current_draft:
            lines = self._current_draft.split("\n")
            for i, line in enumerate(lines):
                if line.startswith(prefix):
                    lines[i] = prefix + action.revision_text
                    self._current_draft = "\n".join(lines)
                    break
            else:
                self._current_draft += f"\n{prefix}{action.revision_text}"
        else:
            self._current_draft = prefix + action.revision_text

        info["revised_section"] = action.section
        return Reward(value=0.0, signals={"intermediate_step": 1.0}, done=False, info=info)
