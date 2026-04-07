"""Task definitions for easy, medium, and hard scenarios."""

from .task_easy import EASY_TASK, grade_easy
from .task_medium import MEDIUM_TASK, grade_medium
from .task_hard import HARD_TASK, grade_hard

TASK_REGISTRY: dict[str, dict] = {
    EASY_TASK["task_id"]: EASY_TASK,
    MEDIUM_TASK["task_id"]: MEDIUM_TASK,
    HARD_TASK["task_id"]: HARD_TASK,
}

GRADER_REGISTRY: dict[str, callable] = {
    EASY_TASK["task_id"]: grade_easy,
    MEDIUM_TASK["task_id"]: grade_medium,
    HARD_TASK["task_id"]: grade_hard,
}

__all__ = [
    "TASK_REGISTRY",
    "GRADER_REGISTRY",
]
