from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


class SolverCode(Enum):
    SCIP = 1
    BRUTER = 2
    ANTCOL = 3


@dataclass
class BaseSolverConfig: ...

SolverConfigType = TypeVar('SolverConfigType', bound=BaseSolverConfig)


@dataclass
class ScipConfig(BaseSolverConfig):
    """
    Configs for ``ScipSolver``.

    :param output_flag: Control console output (internal logs from PySCIPopt) behavior.
        Default: ``False``.
    :param strict_opt: Flag to enforce strict optimal output (disallow NoSolution return, force to raise ``OptimizationError`` instead).
        Default: ``False``.
    """

    output_flag: bool | None = None
    """Flag to control console output (internal logs from PySCIPopt) behavior. Default: ``False``."""
    strict_opt: bool | None = None
    """Flag to enforce strict optimal output (disallow NoSolution return). Default: ``False``."""

    def __post__init__(self):
        msg = []

        if self.output_flag is None:
            self.output_flag = False
        else:
            try:
                self.output_flag = bool(self.output_flag)
            except (TypeError, ValueError):
                msg.append(
                    f"output_flag cannot be converted to bool, "
                    f"given: {self.output_flag!r}, {type(self.output_flag)}",
                )

        if self.strict_opt is None:
            self.strict_opt = False
        else:
            try:
                self.strict_opt = bool(self.strict_opt)
            except (TypeError, ValueError):
                msg.append(
                    f"strict_opt cannot be converted to bool, "
                    f"given: {self.strict_opt!r}, {type(self.strict_opt)}",
                )

        if msg:
            msg = "\n" + "\n".join(msg)
            raise ValueError(msg)
