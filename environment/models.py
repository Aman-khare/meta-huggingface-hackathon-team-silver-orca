"""Pydantic v2 models for the Clinical Note Scribe environment."""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    transcript: str = Field(..., description="Full doctor–patient transcript.")
    task_id: str = Field(..., description="Unique task identifier.")
    patient_context: dict[str, Any] = Field(default_factory=dict, description="Patient demographics and history.")
    current_draft: Optional[str] = Field(None, description="Most recent SOAP-note draft.")
    errors_so_far: list[str] = Field(default_factory=list, description="Accumulated error messages.")
    step_count: int = Field(0, ge=0, description="Steps taken in the current episode.")


class SOAPNote(BaseModel):
    subjective: str = Field(..., description="Patient's self-reported symptoms and history.")
    objective: str = Field(..., description="Clinician's measurable findings.")
    assessment: str = Field(..., description="Clinician's diagnosis or differential.")
    plan: str = Field(..., description="Treatment plan, follow-ups, and prescriptions.")


class Action(BaseModel):
    action_type: Literal["submit_note", "request_clarify", "revise_section"] = Field(..., description="Action kind.")
    soap_note: Optional[SOAPNote] = Field(None, description="SOAP note (required for submit_note).")
    section: Optional[Literal["S", "O", "A", "P"]] = Field(None, description="Section to revise.")
    revision_text: Optional[str] = Field(None, description="Replacement text for the section.")
    clarify_question: Optional[str] = Field(None, description="Clarification question.")


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Aggregate reward [0, 1].")
    signals: dict[str, float] = Field(default_factory=dict, description="Reward sub-signals.")
    done: bool = Field(..., description="Whether the episode ended.")
    info: dict[str, Any] = Field(default_factory=dict, description="Auxiliary metadata.")


class EnvironmentState(BaseModel):
    task_id: str = Field(..., description="Active task identifier.")
    step_count: int = Field(0, ge=0, description="Steps taken so far.")
    max_steps: int = Field(10, ge=1, description="Max steps per episode.")
    done: bool = Field(False, description="Whether the episode terminated.")
    current_draft: Optional[str] = Field(None, description="Latest SOAP draft text.")
    errors_so_far: list[str] = Field(default_factory=list, description="Error messages.")
    last_reward: Optional[Reward] = Field(None, description="Most recent reward.")
    observation: Optional[Observation] = Field(None, description="Most recent observation.")
