from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


class SolverCode(Enum):
    """
    Current solver codes:
        - SCIP
        - BRUTER
        - ANTCOL
    """

    SCIP = 1
    BRUTER = 2
    ANTCOL = 3


@dataclass
class BaseSolverConfig: ...


SolverConfigType = TypeVar("SolverConfigType", bound=BaseSolverConfig)


@dataclass
class ScipConfig(BaseSolverConfig):
    """
    Configs for ``ScipSolver``.

    :param verbose_output: Control console output (internal logs from PySCIPopt) behavior. Used for debugging and optimization process monitoring.
        Default: ``False``.
    """

    verbose_output: bool | None = None
    """
    Control console output (internal logs from PySCIPopt) behavior. Used for debugging and optimization process monitoring.
        Default: ``False``.
    """

    def __post__init__(self):
        msg = []

        if self.verbose_output is None:
            self.verbose_output = False
        else:
            try:
                self.verbose_output = bool(self.verbose_output)
            except (TypeError, ValueError):
                msg.append(
                    f"output_flag cannot be converted to bool, "
                    f"given: {self.verbose_output!r}, {type(self.verbose_output)}",
                )

        if msg:
            msg = "\n" + "\n".join(msg)
            raise ValueError(msg)
