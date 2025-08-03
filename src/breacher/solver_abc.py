from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self, TypeVar

from core import SolverResult, Task

from .solvers_configs import SolverCode, SolverConfigType

# ==========================================================================================
# To add new solver type:
# 1) Implement class in src/breacher/solvers (duh) + decorate with @register_solver
# 2) Add config dataclass in src/breacher/solver_configs.py + update SolverCode enum
# 3) Add overload for get_solver in src/breacher/solver_registry
# 4) Update exports $ imports:
#       src/breacher/solvers/__init__.py (export new solver class)
#       src/breacher/__init__.py (export new config dataclass)
#       src/breacher/solver_registry.py (import new solver code)
# ==========================================================================================


existing_solvers: dict[SolverCode, Callable] = {}


class OptimizationError(Exception):
    """Raised when optimization (solving task) is not possible"""


class Solver[SolverConfigType](ABC):
    def __call__(self, task: Task, config: SolverConfigType) -> tuple[SolverResult, float]:
        """
        Shortcut for ``solver.solve()``
        """
        return self.solve(task, config)

    @abstractmethod
    def solve(self, task: Task, config: SolverConfigType) -> tuple[SolverResult, float]:
        """
        ``Solver.solve()`` is main method to solve task.
        supports ``.__call__()`` shortcuts to this method, look __doc__ of subclass ``.solve()`` method.
        """
        ...


class SeedableSolver(Solver, ABC):
    """
    Abstract base class for solvers that require random number generation.
    """

    @abstractmethod
    def seed(self, value: int) -> Self:
        """Set the random number generator seed."""
        ...


SolverType = TypeVar('SolverType', bound=Solver)


def register_solver(*codes: SolverCode) -> Callable[[type[SolverType]], type[SolverType]]:
    def decorator(cls: type[SolverType]) -> type[SolverType]:
        for code in codes:
            existing_solvers[code] = cls
        return cls
    return decorator
