"""Pydantic v2 models for the Clinical Note Scribe environment.

Defines the typed contracts for observations, actions, rewards,
and overall environment state used by the OpenEnv spec.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation — what the agent sees after each step
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Snapshot of the environment returned to the agent."""

    transcript: str = Field(
        ...,
        description="Full doctor–patient transcript for the current task.",
    )
    task_id: str = Field(
        ...,
        description="Unique identifier for the task (e.g. 'easy_routine_checkup').",
    )
    patient_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured patient demographics and history.",
    )
    current_draft: Optional[str] = Field(
        default=None,
        description="The agent's most recent SOAP-note draft, if any.",
    )
    errors_so_far: list[str] = Field(
        default_factory=list,
        description="Accumulated error/feedback messages from prior steps.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken in the current episode.",
    )


# ---------------------------------------------------------------------------
# SOAPNote — structured clinical note
# ---------------------------------------------------------------------------

class SOAPNote(BaseModel):
    """Standard SOAP clinical-note format."""

    subjective: str = Field(
        ...,
        description="Patient's self-reported symptoms and history.",
    )
    objective: str = Field(
        ...,
        description="Clinician's measurable findings (vitals, exam, labs).",
    )
    assessment: str = Field(
        ...,
        description="Clinician's diagnosis or differential.",
    )
    plan: str = Field(
        ...,
        description="Treatment plan, follow-ups, and prescriptions.",
    )


# ---------------------------------------------------------------------------
# Action — what the agent can do
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """An action the agent submits to the environment."""

    action_type: Literal["submit_note", "request_clarify", "revise_section"] = Field(
        ...,
        description="The kind of action the agent is taking.",
    )

    # --- submit_note fields ---
    soap_note: Optional[SOAPNote] = Field(
        default=None,
        description="Complete SOAP note (required when action_type == 'submit_note').",
    )

    # --- revise_section fields ---
    section: Optional[Literal["S", "O", "A", "P"]] = Field(
        default=None,
        description="Which SOAP section to revise (required when action_type == 'revise_section').",
    )
    revision_text: Optional[str] = Field(
        default=None,
        description="Replacement text for the specified section.",
    )

    # --- request_clarify fields ---
    clarify_question: Optional[str] = Field(
        default=None,
        description="Free-text question the agent asks for clarification.",
    )


# ---------------------------------------------------------------------------
# Reward — multi-signal feedback
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Reward returned after each step."""

    value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate reward in the range [0.0, 1.0].",
    )
    signals: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of individual reward sub-signals.",
    )
    done: bool = Field(
        ...,
        description="Whether the episode has ended.",
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata (e.g. grader diagnostics).",
    )


# ---------------------------------------------------------------------------
# EnvironmentState — full internal state exposed by state()
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Complete snapshot of the environment's internal state."""

    task_id: str = Field(
        ...,
        description="Active task identifier.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Steps taken so far in this episode.",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        description="Maximum steps allowed per episode.",
    )
    done: bool = Field(
        default=False,
        description="Whether the current episode has terminated.",
    )
    current_draft: Optional[str] = Field(
        default=None,
        description="Latest SOAP-note draft text, if any.",
    )
    errors_so_far: list[str] = Field(
        default_factory=list,
        description="Accumulated feedback/error messages.",
    )
    last_reward: Optional[Reward] = Field(
        default=None,
        description="Most recent reward object, if a step has been taken.",
    )
    observation: Optional[Observation] = Field(
        default=None,
        description="Most recent observation returned to the agent.",
    )
