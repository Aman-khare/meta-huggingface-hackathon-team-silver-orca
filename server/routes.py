"""FastAPI routes for the Clinical Note Scribe environment."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from environment.models import Action, EnvironmentState, Observation, Reward
from environment.env import ClinicalNoteScribeEnv

logger = logging.getLogger("clinical_note_scribe.server")
_env = ClinicalNoteScribeEnv()
router = APIRouter()


def _log(event: str, **kw: Any) -> None:
    logger.info(json.dumps({"event": event, "timestamp": time.time(), **kw}, default=str))


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Task to load. Defaults to first registered task.")


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"


@router.post("/reset", response_model=Observation, summary="Reset and start a new episode")
async def reset(body: Optional[ResetRequest] = None) -> Observation:
    task_id = body.task_id if body else None
    _log("START", endpoint="/reset", task_id=task_id)
    try:
        return _env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/step", response_model=StepResponse, summary="Submit an action")
async def step(payload: dict[str, Any]) -> StepResponse:
    try:
        action = Action(**payload)
    except (ValidationError, TypeError) as exc:
        _log("STEP", endpoint="/step", action_type="invalid", error=str(exc))
        error_msg = f"Invalid action payload: {exc}"
        _env._errors_so_far.append(error_msg)
        _env._step_count += 1
        return StepResponse(
            observation=_env._obs(),
            reward=Reward(value=0.0, signals={"error": 1.0}, done=False, info={"error": error_msg}),
            done=False, info={"error": error_msg},
        )

    _log("STEP", endpoint="/step", action_type=action.action_type)
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if done:
        _log("END", endpoint="/step", final_score=reward.value)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@router.get("/state", response_model=EnvironmentState, summary="Inspect environment state")
async def state() -> EnvironmentState:
    return _env.state()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse()
