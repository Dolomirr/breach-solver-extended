# core/struct/__init__.py

from .soft_task import SoftTask
from .solution import NoSolution, Solution, SolverResult
from .task import Task

__all__ = [
    "NoSolution",
    "SoftTask",
    "Solution",
    "SolverResult",
    "Task",
]
