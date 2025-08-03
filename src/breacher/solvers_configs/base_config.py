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

