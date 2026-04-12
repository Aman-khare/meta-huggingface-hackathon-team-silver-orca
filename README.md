# Clinical Note Scribe
An **OpenEnv-compliant** environment for evaluating AI agents on clinical SOAP note generation from doctor patient transcripts.

Built for the **Meta × Hugging Face OpenEnv Hackathon**.

# Clinical Note Scribe
Medical documentation is one of the most time-consuming parts of a doctor's day. After every patient visit, clinicians spend significant time converting spoken conversations into structured clinical notes 
time that could be spent on patient care instead.
Clinical Note Scribe is a reinforcement learning training environment where an AI agent learns to do exactly that: listen to a doctor patient conversation and produce a well structured, accurate, and safe clinical note in SOAP format.
# Interesting part 
The interesting part about this project is that we have created a frontend which shows live what number of reward is given on the basis of submission.

Three levels of difficulty
Tasks range from a routine check-up all the way to a chaotic ER visit with overlapping symptoms and urgent orders each with its own grader and reward logic.


---

## Environment Description

A doctor–patient conversation is recorded as a text transcript. The agent's goal is to read the transcript along with structured patient context (demographics, medications, labs) and produce a clinically accurate, concise **SOAP note** (Subjective, Objective, Assessment, Plan).

The agent interacts through a standard `reset()` / `step()` / `state()` API. Three action types are available: submit a full note, request clarification, or revise a single section. A multi-signal reward function scores each submission on clinical accuracy, conciseness, safe language, and structural validity, with penalties for excessive steps or invalid actions.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `transcript` | `str` | Full doctor–patient transcript for the current task |
| `task_id` | `str` | Unique identifier for the active task |
| `patient_context` | `dict[str, Any]` | Structured patient demographics, conditions, medications, allergies, and labs |
| `current_draft` | `Optional[str]` | The agent's most recent SOAP-note draft (null until first submission or revision) |
| `errors_so_far` | `list[str]` | Accumulated error/feedback messages from prior invalid actions |
| `step_count` | `int` | Number of steps taken so far in the current episode (0-indexed at reset) |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | `Literal["submit_note", "request_clarify", "revise_section"]` | **Required.** The kind of action the agent is taking |
| `soap_note` | `Optional[SOAPNote]` | Complete SOAP note — **required** when `action_type == "submit_note"` |
| `section` | `Optional[Literal["S", "O", "A", "P"]]` | Which SOAP section to revise — **required** when `action_type == "revise_section"` |
| `revision_text` | `Optional[str]` | Replacement text for the section — **required** when `action_type == "revise_section"` |
| `clarify_question` | `Optional[str]` | Free-text question — **required** when `action_type == "request_clarify"` |

### SOAPNote Schema

| Field | Type | Description |
|---|---|---|
| `subjective` | `str` | Patient's self-reported symptoms, history, and concerns |
| `objective` | `str` | Clinician's measurable findings — vitals, exam, labs, imaging |
| `assessment` | `str` | Differential diagnoses and clinical reasoning |
| `plan` | `str` | Treatment plan, medications, follow-ups, referrals |

---

## Tasks

### 🟢 Easy — Routine Check-Up
**Task ID:** `easy_routine_checkup` · **Max steps:** 5

A 6-turn dialogue about a common cold and blood pressure screening for a 34-year-old female. Straightforward clinical picture with no complications.

### 🟡 Medium — Chronic Disease Follow-Up
**Task ID:** `medium_chronic_disease_followup` · **Max steps:** 8

A 14-turn follow-up visit for a 58-year-old male with Type 2 Diabetes and Hypertension. Includes HbA1c lab review (7.2% → 7.8%), medication adjustments (adding glipizide 5 mg, uptitrating lisinopril 20 → 40 mg), a 2-week statin gap, and dietary counselling around restaurant meals.

### 🔴 Hard — Complex ER Visit
**Task ID:** `hard_complex_er_visit` · **Max steps:** 10

A rapid 20-turn emergency-room encounter for a 72-year-old female with CAD, AFib, and CKD Stage 3. Overlapping chest pain and shortness of breath with a dual ACS vs PE differential. Includes a patient self-contradiction (denied then admitted nitroglycerin use at home), contrast dye allergy complicating CT-PA workup (V/Q scan ordered instead), elevated D-dimer (1840 ng/mL), and Cardiac ICU admission.

---

## Reward Function

```
value = clamp(weighted_sum − deductions, 0.0, 1.0)
```

| Signal | Weight | Criteria |
|---|---|---|
| `grader_score` | × 0.60 | Clinical accuracy from task-specific grader |
| `conciseness_bonus` | × 0.10 | 1.0 if total SOAP note ≤ 400 words |
| `safe_language_score` | × 0.15 | 1.0 if no unsafe-certainty phrases detected |
| `format_valid` | × 0.15 | 1.0 if all four SOAP fields are non-empty |

| Deduction | Rate | Trigger |
|---|---|---|
| Step penalty | −0.05 | Per step beyond 3 (penalises excessive clarification) |
| Error penalty | −0.10 | Per invalid action in `errors_so_far` |

---

## Installation

### Prerequisites

- Python 3.11+
- An OpenAI-compatible API key (set as `HF_TOKEN`)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/meta-huggingface-hackathon-team-silver-orca.git
cd meta-huggingface-hackathon-team-silver-orca

# Create a virtual environment (optional but recommended)
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
docker build -t meta-huggingface-hackathon-team-silver-orca .
```

---

## Usage

### 1. Start the Environment Server

The environment runs as a REST API. Start the server first before running the agent.

#### Using Python
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

#### Using Docker
```bash
docker run -p 7860:7860 meta-huggingface-hackathon-team-silver-orca
```

### 2. Run the Agent (Inference)

In another terminal, run the baseline inference script which will interact with the running environment:

**On Linux/macOS:**
```bash
export HF_TOKEN="sk-..."
export MODEL_NAME="gpt-4o-mini"           # or any OpenAI-compatible model
export API_BASE_URL="https://api.openai.com/v1"
python inference.py
```

**On Windows (PowerShell):**
```powershell
$env:HF_TOKEN="sk-..."
$env:MODEL_NAME="gpt-4o-mini"
$env:API_BASE_URL="https://api.openai.com/v1"
python inference.py
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe → `{"status": "ok"}` |
| `POST` | `/reset` | Start a new episode → `Observation` |
| `POST` | `/step` | Submit an action → `{observation, reward, done, info}` |
| `GET` | `/state` | Inspect environment state → `EnvironmentState` |
---

## Baseline Scores

Scores obtained using `gpt-4o-mini` with `temperature=0.2` via `inference.py`:

| Task | Difficulty | Score |
|---|---|---|
| `easy_routine_checkup` | 🟢 Easy | 0.8520 |
| `medium_chronic_disease_followup` | 🟡 Medium | 0.7450 |
| `hard_complex_er_visit` | 🔴 Hard | 0.5110 |
| **Average** | | **0.7026** |

> **Note:** These baseline scores use dynamic clinical graders that check for explicit diagnoses and strict formatting. Scores will accurately vary based on the specific LLM used.

---

## Structured Logging

Every episode emits JSON log lines to stdout, scraped by the OpenEnv validator:

```json
{"event": "START", "task_id": "easy_routine_checkup", "timestamp": 1700000000.0}
{"event": "STEP",  "step": 1, "action_type": "submit_note", "reward": 0.82}
{"event": "END",   "task_id": "easy_routine_checkup", "final_score": 0.82}
```

---

## Project Structure

```
meta-huggingface-hackathon-team-silver-orca/
├── openenv.yaml              ← OpenEnv spec metadata + graders
├── inference.py              ← Baseline inference (OpenAI client, all 3 tasks)
├── Dockerfile                ← Containerised server (port 7860)
├── README.md                 ← This file
├── requirements.txt
│
├── environment/
│   ├── __init__.py
│   ├── models.py             ← Pydantic v2 models (Observation, Action, Reward, …)
│   ├── env.py                ← ClinicalNoteScribeEnv (reset/step/state)
│   ├── reward.py             ← Multi-signal reward function
│   └── tasks/
│       ├── __init__.py       ← Task & grader registries
│       ├── task_easy.py      ← Routine check-up + grader stub
│       ├── task_medium.py    ← Chronic disease follow-up + grader stub
│       └── task_hard.py      ← Complex ER visit + grader stub
│
├── server/
│   ├── __init__.py
│   ├── app.py                ← FastAPI application
│   └── routes.py             ← API route definitions
│
└── data/
    ├── transcripts/
    │   ├── easy.txt           ← 6-turn routine check-up transcript
    │   ├── medium.txt         ← 14-turn chronic disease follow-up transcript
    │   └── hard.txt           ← 20-turn complex ER visit transcript
    └── clarify_answers.json   ← Clarification Q&A lookup (10 entries)
```

---

## License

MIT
