import random as rand
from functools import cached_property

import numpy as np
from pyscipopt import Expr, Model, Variable

from core import HexSymbol, Task

from ...solvers_configs import ScipConfig


class TaskContext:
    """
    A class representing the context of a task in a breach protocol.

    Attributes
    ----------
        y: list[str]
            A list of scip.Variable labels each variable indicates whether daemon i is activated.
        z: list[dict[int, Variable]]
            Contains position where each daemon can start and whether this starting position is valid.
        buffer_seq: list[Expr]
            Value expression with number chosen in corresponding buffer sequence.
        used_buffer: Expr
            Number of filled buffer slots.
        n: int
            Returns the number of rows in the ``x`` matrix.
        m: int
            Returns the length of each row in the ``x`` matrix.
        d_count: int
            Returns the number of daemons in the task.
        d_lengths: np.ndarray[tuple[int] np.dtype[np.int64]]
            Returns the lengths of daemons after 'stripping from padding with ``HexSymbol.S_STOP``'.
        unused_cell_reward: np.float64
            Calculates the reward per unused buffer slot.

    """

    x: dict[tuple[int, int, int], str]
    """3d binary matrix of scip.Variable labels each cell of (n, m, t) indicates whether cell (n, m) from task.matrix is chosen in step t"""
    y: list[str]
    """list of scip.Variable labels each variable indicates wether daemon i is activated"""
    z: list[dict[int, Variable]]
    """contains position where each daemon can starts and whether this starting position is valid"""
    buffer_seq: list[Expr]
    """value expression with number chosen in corresponding buffer sequence"""
    used_buffer: Expr

    def __init__(self, task: Task, config: ScipConfig):
        self.matrix = task.matrix
        self.daemons = task.daemons
        self.daemons_costs = task.daemons_costs
        self.buffer_size = task.buffer_size

        self.config = config
        self.model = Model(f"BreachProtocol_{rand.randint(0, 99999)}")

        self.x = {}
        self.y = []
        self.z = []
        self.buffer_seq = []

        self.is_finished: bool = False

    @cached_property
    def n(self) -> int:
        """Number of rows in the ``x`` matrix."""
        return self.matrix.shape[0]

    @cached_property
    def m(self) -> int:
        """Length of each row in the ``x`` matrix."""
        return self.matrix.shape[1]

    @cached_property
    def d_count(self) -> int:
        """Number of daemons in task."""
        return self.daemons.shape[0]

    @cached_property
    def d_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Length of daemons after 'stripping from padding with ``HexSymbol.S_STOP``."""
        return np.sum(self.daemons != HexSymbol.S_STOP, axis=1)

    @cached_property
    def unused_cell_reward(self) -> np.float64:
        """
        Reward per unused buffer slot,
        calculated in a way to ensure that activating any new daemon is more rewarding than preserving buffer.
        """
        try:
            rewards_per_symbol = self.daemons_costs / self.d_lengths
        except (RuntimeWarning, ZeroDivisionError):
            return np.float64(0.0)
        return 0.1 * rewards_per_symbol.min()

