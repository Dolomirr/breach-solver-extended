from typing import Literal, overload

from .solver_abc import Solver, existing_solvers
from .solver_configs import SolverCode
from .solvers import ScipSolver


class GetSolver:
    """
    Allow to get Solver subclass instance by code.
    
    Provide such methods:
        - ``GetSolver.single()`` to retrieve single instance of specific subclass (recommended due to typecasting)
        - ``GetSolver.multiple()`` to create dict with instances of multiple subclasses.
        - ``GetSolver.all()`` to create dict with instances of all existing subclasses.
    
    """
    
    @overload
    @staticmethod
    def single(code: Literal[SolverCode.SCIP]) -> ScipSolver: ...

    @overload
    @staticmethod
    def single(code: Literal[SolverCode.BRUTER]) -> ScipSolver: ... # TODO: add correct overload
    
    @overload
    @staticmethod
    # fallback
    def single(code: SolverCode) -> Solver: ...

    @staticmethod
    def single(code: SolverCode) -> Solver:
        """
        :returns: single instance of specific subclass.
        """
        cls = existing_solvers.get(code)
        if cls is None:
            #? unreachable?
            msg = f"Unknown solver: {code}, must be one of {list(existing_solvers.keys())}"
            raise ValueError(msg)
        return cls()
    
    @staticmethod
    def multiple(*codes: SolverCode) -> dict[SolverCode, Solver]:
        """
        :returns: dict with solver instances for each provided solver code.
        """
        solvers = {}
        for code in codes:
            solvers[code] = GetSolver.single(code)
        return solvers
    
    @staticmethod
    def all() -> dict[SolverCode, Solver]:
        """
        :returns: dict with instance of all Solver subclasses.
        """
        return GetSolver.multiple(*SolverCode)
