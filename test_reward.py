import sys
sys.path.insert(0, ".")

from environment import ClinicalNoteScribeEnv, Action, SOAPNote
from environment.reward import (
    compute_reward, _conciseness_bonus, _safe_language_score, _format_valid,
    WORD_LIMIT, FREE_STEPS, STEP_PENALTY_RATE, ERROR_PENALTY_RATE,
)

def check(label, got, want):
    ok = abs(got - want) < 1e-6
    sym = "OK" if ok else "FAIL"
    print(f"  [{sym}] {label}: got={got}  want={want}")
    return ok

short_note = SOAPNote(
    subjective="Headache and runny nose for 5 days.",
    objective="BP 118/76, HR 72, afebrile, clear lungs.",
    assessment="Viral URI.",
    plan="DayQuil, fluids, rest. Follow up if fever develops.",
)
long_note  = SOAPNote(subjective=" ".join(["word"] * (WORD_LIMIT + 1)), objective="O", assessment="A", plan="P")
unsafe_note = SOAPNote(subjective="Patient definitely has pneumonia.", objective="O", assessment="A", plan="P")
empty_note  = SOAPNote(subjective="", objective="O", assessment="A", plan="P")

submit_ok  = Action(action_type="submit_note", soap_note=short_note)
submit_bad = Action(action_type="submit_note", soap_note=empty_note)
clarify    = Action(action_type="request_clarify", clarify_question="fever?")

print("\n--- Sub-signal unit tests ---")
check("conciseness(short)", _conciseness_bonus(short_note), 1.0)
check("conciseness(long) ", _conciseness_bonus(long_note),  0.0)
check("safe_lang(clean)  ", _safe_language_score(short_note),  1.0)
check("safe_lang(unsafe) ", _safe_language_score(unsafe_note), 0.0)
check("format_valid(ok)  ", _format_valid(submit_ok),  1.0)
check("format_valid(bad) ", _format_valid(submit_bad), 0.0)
check("format_valid(clfy)", _format_valid(clarify),    1.0)

print("\n--- grader=1.0, steps=2, errors=0 → expect value=1.0 ---")
r = compute_reward(submit_ok, grader_score=1.0, step_count=2, errors_so_far=[])
check("value            ", r.value, 1.0)
check("grader_score wt  ", r.signals["grader_score"],         0.60)
check("conciseness wt   ", r.signals["conciseness_bonus"],    0.10)
check("safe_lang wt     ", r.signals["safe_language_score"],  0.15)
check("format_valid wt  ", r.signals["format_valid"],         0.15)
check("step_penalty     ", r.signals["step_penalty"],         0.0)
check("error_penalty    ", r.signals["error_penalty"],        0.0)

print("\n--- grader=1.0, steps=5 (+2 extra) → expect deduct 0.10 ---")
r2 = compute_reward(submit_ok, grader_score=1.0, step_count=5, errors_so_far=[])
check("step_penalty     ", r2.signals["step_penalty"], -(2 * STEP_PENALTY_RATE))
check("value            ", r2.value, round(1.0 - 2 * STEP_PENALTY_RATE, 4))

print("\n--- grader=1.0, steps=2, errors=2 → expect deduct 0.20 ---")
r3 = compute_reward(submit_ok, grader_score=1.0, step_count=2, errors_so_far=["e1", "e2"])
check("error_penalty    ", r3.signals["error_penalty"], -(2 * ERROR_PENALTY_RATE))
check("value            ", r3.value, round(1.0 - 2 * ERROR_PENALTY_RATE, 4))

print("\n--- all bad signals → expect value clamped to 0.0 ---")
bad_note = SOAPNote(subjective=" ".join(["word"] * 500) + " Patient definitely has cancer.", objective="", assessment="A", plan="P")
bad_act  = Action(action_type="submit_note", soap_note=bad_note)
r4 = compute_reward(bad_act, grader_score=0.0, step_count=10, errors_so_far=["e1","e2","e3"])
check("value clamped    ", r4.value, 0.0)

print("\n--- end-to-end env: clarify(step1) then submit(step2) ---")
env = ClinicalNoteScribeEnv()
env.reset("easy_routine_checkup")
_, rc, dc, _ = env.step(Action(action_type="request_clarify", clarify_question="did the patient report any fever?"))
check("clarify done=False", float(dc), 0.0)
_, rs, ds, _ = env.step(submit_ok)
check("submit  done=True ", float(ds), 1.0)
assert 0.0 <= rs.value <= 1.0
print(f"  Final value: {rs.value}")
print(f"  Signals: { {k:v for k,v in rs.signals.items() if not k.startswith('_')} }")
print("\nAll done.")
