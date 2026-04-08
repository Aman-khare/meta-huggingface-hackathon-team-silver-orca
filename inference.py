"""
Inference Script — Clinical Note Scribe
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   The name of the local image to use for the environment
                       if you are using from_docker_image() method.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after the task finishes, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1].

Designed to complete in under 20 minutes on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Silence the underlying env's stdout JSON logs (redirect them to stderr)
# ---------------------------------------------------------------------------
env_logger = logging.getLogger("clinical_note_scribe")
env_logger.setLevel(logging.INFO)
env_logger.handlers.clear()
env_logger.addHandler(logging.StreamHandler(sys.stderr))
env_logger.propagate = False


# ---------------------------------------------------------------------------
# Environment imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import ClinicalNoteScribeEnv, Action, SOAPNote  # noqa: E402
from environment.tasks import TASK_REGISTRY                       # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN         = os.getenv("HF_TOKEN")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK    = "clinical-note-scribe"
TASK_IDS     = list(TASK_REGISTRY.keys())
MAX_STEPS    = 5       # Max steps per task (submit + optional clarify/revise)
MAX_TOKENS   = 1024
TEMPERATURE  = 0.2

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical documentation assistant. Given a doctor-patient transcript
    and patient context, generate a concise, clinically accurate SOAP note.

    RULES:
    1. Use professional medical language. Avoid over-certain phrasing such as
       "patient definitely has", "diagnosis is certain", or "100% certain".
    2. Keep the note concise - aim for under 400 words total across all four sections.
    3. Return your output as a **single valid JSON object** matching this schema exactly:

    {
      "action_type": "submit_note",
      "soap_note": {
        "subjective": "<patient's reported symptoms, history, and concerns>",
        "objective": "<exam findings, vitals, lab results, imaging>",
        "assessment": "<differential diagnoses and clinical reasoning>",
        "plan": "<treatment plan, medications, follow-up, referrals>"
      }
    }

    Return ONLY the JSON object. No markdown fences, no commentary, no extra keys.
""").strip()


# ---------------------------------------------------------------------------
# Stdout logging — mandatory hackathon format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(transcript: str, patient_context: dict[str, Any]) -> str:
    """Build the user message containing the transcript and context."""
    ctx_str = json.dumps(patient_context, indent=2, default=str)
    return (
        f"## Patient Context\n```json\n{ctx_str}\n```\n\n"
        f"## Doctor-Patient Transcript\n```\n{transcript}\n```\n\n"
        "Generate the SOAP note as a JSON Action object."
    )


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse the model's raw text output into a dict, tolerating markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"[DEBUG] Failed to parse model output as JSON: {exc}", file=sys.stderr, flush=True)
        print(f"[DEBUG] Raw output:\n{raw}", file=sys.stderr, flush=True)
        raise


def get_soap_note(client: OpenAI, transcript: str, patient_context: dict[str, Any]) -> dict[str, Any]:
    """Call the OpenAI-compatible API and return the parsed JSON action dict."""
    user_prompt = _build_user_prompt(transcript, patient_context)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return _parse_json(raw)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        raise


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, env: ClinicalNoteScribeEnv, task_id: str) -> dict[str, Any]:
    """Run a single task episode and return the result dict."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ---- reset ----
        obs = env.reset(task_id)

        for step in range(1, MAX_STEPS + 1):
            # ---- generate SOAP note via LLM ----
            try:
                action_dict = get_soap_note(client, obs.transcript, obs.patient_context)
                action = Action(**action_dict)
                action_str = f"submit_note(sections=S,O,A,P)"
            except Exception as exc:
                # On model / parse failure, submit an empty note so all sub-signals
                # grade to 0.0 (format_valid=0 because fields are empty, grader=0).
                action = Action(
                    action_type="submit_note",
                    soap_note=SOAPNote(
                        subjective="",
                        objective="",
                        assessment="",
                        plan="",
                    ),
                )
                action_str = "submit_note(fallback)"
                last_error = str(exc)

            # ---- step ----
            obs, reward_obj, done, info = env.step(action)

            reward_val = reward_obj.value
            rewards.append(reward_val)
            steps_taken = step

            # Check for env-level errors
            error_msg = None
            if obs.errors_so_far:
                error_msg = obs.errors_so_far[-1]
            elif last_error:
                error_msg = last_error
                last_error = None

            log_step(
                step=step,
                action=action_str,
                reward=reward_val,
                done=done,
                error=error_msg,
            )

            if done:
                break

        # Final score = last reward value (already in [0, 1])
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", file=sys.stderr, flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print(
            "[DEBUG] WARNING: HF_TOKEN is not set. "
            "Model calls will fail unless the endpoint requires no auth.",
            file=sys.stderr,
            flush=True,
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ClinicalNoteScribeEnv()
    results: List[dict[str, Any]] = []

    for task_id in TASK_IDS:
        result = run_task(client, env, task_id)
        results.append(result)

    # ---- Summary table ----
    print("", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    print("  SUMMARY", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)

    col_task = max(len("Task"), *(len(r["task_id"]) for r in results))
    header = f"  {'Task':<{col_task}}  {'Score':>7}  {'Steps':>5}"
    sep    = f"  {'-' * col_task}  {'-' * 7}  {'-' * 5}"
    print(header, file=sys.stderr, flush=True)
    print(sep, file=sys.stderr, flush=True)

    total_score = 0.0
    for r in results:
        s = f"{r['score']:.4f}" if r["success"] else "ERROR"
        print(f"  {r['task_id']:<{col_task}}  {s:>7}  {r['steps']:>5}", file=sys.stderr, flush=True)
        total_score += r["score"]

    print(sep, file=sys.stderr, flush=True)
    avg = total_score / len(results) if results else 0.0
    print(f"  {'AVERAGE':<{col_task}}  {avg:>7.4f}", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
