import random as rand
from functools import cached_property, wraps

import numpy as np
from pyscipopt import Expr, Model, Variable

from core import HexSymbol, Task

from ...solver_configs import ScipConfig


def require_finished(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, 'is_finished', False):
            msg = f"{method.__name__} requires ModelRunner.optimize() to complete first"
            raise RuntimeError(msg)
        return method(self, *args, **kwargs)
    return wrapper


class TaskContext:
    x: dict[tuple[int, int, int], str]
    """
    3d matrix of scip.Variable labels each cell of (n, m, t) indicates whether cell (n, m) from task.matrix is chosen in step t
    """
    y: list[str]
    """
    list of scip.Variable labels each variable indicates wether daemon i is activated
    """
    z: list[dict[int, Variable]]
    """
    contains position where each daemon can starts and whether this starting position is valid
    """
    buffer_seq: list[Expr]
    """
    value expression with number chosen in corresponding buffer sequence
    """
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

    # TODO: n x m
    @cached_property
    def n(self) -> int:
        return self.matrix.shape[0]
    
    @cached_property
    def m(self) -> int:
        return self.matrix.shape[1]

    @cached_property
    def d_count(self) -> int:
        return self.daemons.shape[0]

    @cached_property
    def d_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        return np.sum(self.daemons != HexSymbol.S_STOP, axis=1)

    @cached_property
    def unused_cell_reward(self) -> np.float64:
        rewards_per_symbol = self.daemons_costs / self.d_lengths
        return 0.1 * rewards_per_symbol.min()

    
    @property
    @require_finished
    def path(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int8]]:
        return np.array(
            [
                (i, j)
                for t in range(self.buffer_size)
                for i in range(self.n)
                for j in range(self.m)
                if self.model.getVal(self.x[i, j, t]) > 0.5
            ],
            dtype=np.int8,
        )

    @property
    @require_finished
    def buffer_nums(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int8]]:
        return np.array(
                [
                    round(self.model.getVal(self.buffer_seq[t]))
                    for t in range(self.buffer_size)
                ],
            )
    
    @property
    @require_finished
    def active_daemons(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int8]]:
        y_active = np.zeros(self.d_count, dtype=bool)
        for i, var in enumerate(self.y):
            y_active[i] = bool(round(self.model.getVal(var)))
        return y_active

    @property
    @require_finished
    def total_points(self) -> np.int64:
        if not self.is_finished:
            msg = 'This property should not be called before model optimization is finished.'
            raise RuntimeError(msg)
        return np.dot(self.daemons_costs, self.active_daemons)
