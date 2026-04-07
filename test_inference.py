import sys
sys.path.insert(0, ".")
from inference import SYSTEM_PROMPT, TASK_IDS, _parse_json, _build_user_prompt
from environment import Action

print("Imports OK")
print("Tasks:", TASK_IDS)

# Test JSON parsing
j = _parse_json('{"action_type": "submit_note", "soap_note": {"subjective": "S", "objective": "O", "assessment": "A", "plan": "P"}}')
print("Parse OK:", j["action_type"])

# Test markdown fence stripping
fenced = '```json\n{"action_type": "submit_note", "soap_note": {"subjective": "S", "objective": "O", "assessment": "A", "plan": "P"}}\n```'
j2 = _parse_json(fenced)
print("Fence strip OK:", j2["action_type"])

# Test Action creation from parsed output
action = Action(**j2)
print("Action created:", action.action_type, "/ sections:", list(action.soap_note.model_fields.keys()))

# Test prompt building
p = _build_user_prompt("Hello doctor", {"name": "Test", "age": 30})
print("Prompt len:", len(p), "chars")

print("\nAll checks passed.")
