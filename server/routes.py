"""FastAPI route definitions for the Clinical Note Scribe environment.

Endpoints
---------
POST /reset   – start a new episode (takes ``task_id``, returns ``Observation``)
POST /step    – execute an action   (takes ``Action``,  returns step result)
GET  /state   – inspect env state   (returns ``EnvironmentState``)
GET  /health  – liveness probe      (returns ``{"status": "ok"}``)

Structured logging
------------------
The underlying ``ClinicalNoteScribeEnv`` already emits ``[START]``, ``[STEP]``,
and ``[END]`` JSON lines to stdout via Python's ``logging`` module.  This router
adds a thin request-level log wrapper so every inbound HTTP call is also
traceable.
"""

from __future__ import annotations

import logging
import json
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from environment.models import (
    Action,
    EnvironmentState,
    Observation,
    Reward,
)
from environment.env import ClinicalNoteScribeEnv

logger = logging.getLogger("clinical_note_scribe.server")


# ---------------------------------------------------------------------------
# Singleton environment instance
# ---------------------------------------------------------------------------

_env = ClinicalNoteScribeEnv()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Task to load. One of: easy_routine_checkup, "
            "medium_chronic_disease_followup, hard_complex_er_visit. "
            "Defaults to the first registered task."
        ),
    )


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(event: str, **kwargs: Any) -> None:
    """Emit a structured JSON log line to stdout."""
    payload: dict[str, Any] = {"event": event, "timestamp": time.time()}
    payload.update(kwargs)
    logger.info(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post(
    "/reset",
    response_model=Observation,
    summary="Reset the environment and start a new episode",
)
async def reset(body: ResetRequest) -> Observation:
    """Load a task and return the initial ``Observation``.

    The underlying environment emits a ``[START]`` log event.
    """
    _log("START", endpoint="/reset", task_id=body.task_id)
    try:
        obs = _env.reset(task_id=body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@router.post(
    "/step",
    response_model=StepResponse,
    summary="Submit an action and advance the environment by one step",
)
async def step(action: Action) -> StepResponse:
    """Execute *action* in the current episode.

    The underlying environment emits a ``[STEP]`` log event (and ``[END]``
    when the episode terminates).
    """
    _log("STEP", endpoint="/step", action_type=action.action_type)
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        # e.g. stepping after episode is done without reset
        raise HTTPException(status_code=409, detail=str(exc))

    if done:
        _log("END", endpoint="/step", final_score=reward.value)

    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@router.get(
    "/state",
    response_model=EnvironmentState,
    summary="Return the full internal environment state",
)
async def state() -> EnvironmentState:
    """Inspect the environment without mutating it."""
    return _env.state()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Returns HTTP 200 with ``{"status": "ok"}``."""
    return HealthResponse()
