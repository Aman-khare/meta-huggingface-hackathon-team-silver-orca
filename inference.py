"""Baseline inference script for the Clinical Note Scribe environment.

Runs all three tasks (easy → medium → hard) sequentially using an
OpenAI-compatible API to generate SOAP notes from doctor–patient transcripts.

Environment variables
---------------------
OPENAI_API_KEY   – API key for the model provider
API_BASE_URL     – Base URL for the OpenAI-compatible endpoint (default: https://api.openai.com/v1)
MODEL_NAME       – Model identifier to use (default: gpt-4o-mini)

Usage::

    python inference.py

Designed to complete in under 20 minutes on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap logging BEFORE importing environment modules so the root logger
# is configured and child loggers (clinical_note_scribe.*) propagate cleanly.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Environment imports (after logging is configured)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import ClinicalNoteScribeEnv, Action, SOAPNote  # noqa: E402
from environment.tasks import TASK_REGISTRY                       # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY      = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_IDS = list(TASK_REGISTRY.keys())  # deterministic order

# Maximum tokens for the model response — keeps latency low
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical documentation assistant. Given a doctor–patient transcript \
and patient context, generate a concise, clinically accurate SOAP note.

RULES:
1. Use professional medical language.  Avoid over-certain phrasing such as \
   "patient definitely has", "diagnosis is certain", or "100% certain".
2. Keep the note concise — aim for under 400 words total across all four sections.
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

Return ONLY the JSON object.  No markdown fences, no commentary, no extra keys.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(transcript: str, patient_context: dict[str, Any]) -> str:
    """Build the user message containing the transcript and context."""
    ctx_str = json.dumps(patient_context, indent=2, default=str)
    return (
        f"## Patient Context\n```json\n{ctx_str}\n```\n\n"
        f"## Doctor–Patient Transcript\n```\n{transcript}\n```\n\n"
        "Generate the SOAP note as a JSON Action object."
    )


def _call_model(user_prompt: str) -> dict[str, Any]:
    """Call the OpenAI-compatible API and return the parsed JSON action dict.

    Uses ``urllib`` so there is zero dependency on ``openai`` package —
    this keeps the Docker image small and avoids version conflicts.
    Falls back to the ``openai`` package if installed.
    """
    try:
        return _call_model_sdk(user_prompt)
    except ImportError:
        return _call_model_urllib(user_prompt)


def _call_model_sdk(user_prompt: str) -> dict[str, Any]:
    """Call via the ``openai`` Python SDK."""
    from openai import OpenAI  # noqa: F811

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_json(raw)


def _call_model_urllib(user_prompt: str) -> dict[str, Any]:
    """Fallback: call the API with ``urllib`` (no extra dependencies)."""
    import urllib.request

    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())

    raw = body["choices"][0]["message"]["content"].strip()
    return _parse_json(raw)


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse the model's raw text output into a dict, tolerating markdown fences."""
    # Strip markdown code fences if present
    cleaned = raw
    if cleaned.startswith("```"):
        # remove opening fence (possibly ```json)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse model output as JSON: %s", exc)
        logger.error("Raw output:\n%s", raw)
        raise


def _log_event(event: str, **kwargs: Any) -> None:
    """Emit a structured JSON log line."""
    payload: dict[str, Any] = {"event": event, "timestamp": time.time()}
    payload.update(kwargs)
    logger.info(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_all_tasks() -> list[dict[str, Any]]:
    """Run every registered task and return a list of result dicts."""
    env = ClinicalNoteScribeEnv()
    results: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  TASK: %s", task_id)
        logger.info("=" * 60)

        t0 = time.time()
        _log_event("INFERENCE_START", task_id=task_id)

        # ---- reset ----
        obs = env.reset(task_id)
        logger.info("  Transcript length : %d chars", len(obs.transcript))
        logger.info("  Patient context keys: %s", list(obs.patient_context.keys()))

        # ---- generate SOAP note via LLM ----
        user_prompt = _build_user_prompt(obs.transcript, obs.patient_context)
        logger.info("  Calling model (%s) ...", MODEL_NAME)

        try:
            action_dict = _call_model(user_prompt)
        except Exception as exc:
            logger.error("  Model call failed: %s", exc)
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "error": str(exc),
                "elapsed_s": round(time.time() - t0, 2),
            })
            _log_event("INFERENCE_ERROR", task_id=task_id, error=str(exc))
            continue

        # ---- validate and create Action ----
        try:
            action = Action(**action_dict)
        except Exception as exc:
            logger.error("  Invalid action schema: %s", exc)
            logger.error("  Model returned: %s", json.dumps(action_dict, indent=2))
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "error": f"schema_error: {exc}",
                "elapsed_s": round(time.time() - t0, 2),
            })
            _log_event("INFERENCE_ERROR", task_id=task_id, error=str(exc))
            continue

        # ---- step (submit) ----
        obs2, reward, done, info = env.step(action)
        elapsed = round(time.time() - t0, 2)

        logger.info("  Done: %s  |  Reward: %.4f  |  Elapsed: %.1fs", done, reward.value, elapsed)
        logger.info("  Signals: %s",
                     {k: v for k, v in reward.signals.items() if not k.startswith("_")})

        _log_event("INFERENCE_END", task_id=task_id, score=reward.value, elapsed_s=elapsed)

        results.append({
            "task_id": task_id,
            "score": reward.value,
            "elapsed_s": elapsed,
        })

    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)

    col_task  = max(len("Task"), *(len(r["task_id"]) for r in results))
    col_score = 7  # "Score" + padding
    col_time  = 9  # "Time (s)"

    header = f"  {'Task':<{col_task}}  {'Score':>{col_score}}  {'Time (s)':>{col_time}}"
    sep    = f"  {'-' * col_task}  {'-' * col_score}  {'-' * col_time}"
    logger.info(header)
    logger.info(sep)

    total_score = 0.0
    for r in results:
        score_str = f"{r['score']:.4f}" if "error" not in r else "ERROR"
        time_str  = f"{r['elapsed_s']:.1f}"
        logger.info(f"  {r['task_id']:<{col_task}}  {score_str:>{col_score}}  {time_str:>{col_time}}")
        total_score += r["score"]

    logger.info(sep)
    avg = total_score / len(results) if results else 0.0
    logger.info(f"  {'AVERAGE':<{col_task}}  {avg:>{col_score}.4f}")
    logger.info("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not API_KEY:
        logger.warning(
            "OPENAI_API_KEY is not set. The model calls will fail unless "
            "the API endpoint does not require authentication."
        )

    logger.info("Clinical Note Scribe — Baseline Inference")
    logger.info("  Model      : %s", MODEL_NAME)
    logger.info("  API Base   : %s", API_BASE_URL)
    logger.info("  Tasks      : %s", TASK_IDS)
    logger.info("")

    start = time.time()
    results = run_all_tasks()
    total_elapsed = round(time.time() - start, 2)

    _print_summary(results)
    logger.info("  Total wall-clock time: %.1fs", total_elapsed)

    _log_event("INFERENCE_COMPLETE", total_elapsed_s=total_elapsed,
               scores={r["task_id"]: r["score"] for r in results})
