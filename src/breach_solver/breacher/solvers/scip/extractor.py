from itertools import product

import numpy as np

from .context import TaskContext


class ResultExtractor:
    """
    Attributes:
        path: Returns sequential coordinates of chosen cells in buffer formatted for directly creating ``Solution`` instance.
        buffer_nums: Returns symbols chosen in buffer formatted for directly creating ``Solution`` instance.
        active_daemons: Returns an array indicating whether each daemon sequence is active formatted for directly creating ``Solution`` instance.
        total_points: Calculates the amount of earned points.

    """

    def __init__(self, context: TaskContext):
        self.context = context

    def _require_finished(self):
        if not self.context.is_finished:
            msg = "requires ModelRunner.optimize() to complete first"
            raise RuntimeError(msg)

    @property
    def path(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int8]]:
        """
        Sequential coordinates of chosen cells in buffer.

        Formatted for directly creating ``Solution`` instance.
        :return: 2d array with shape (n, 2)
        """
        self._require_finished()
        return np.array(
            [
                (i, j)
                for t, i, j in product(range(self.context.buffer_size), range(self.context.n), range(self.context.m))
                if self.context.model.getVal(self.context.x[i, j, t]) > 0.5
            ],
            dtype=np.int8,
        )

    @property
    def buffer_nums(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int8]]:
        """
        Symbols (corresponds to  ``HexSymbols`` enum values) chosen in buffer.

        Formatted for directly creating ``Solution`` instance.
        :return: 1d array, can be shorter then buffer_size
        """
        self._require_finished()
        return np.array(
            [round(self.context.model.getVal(self.context.buffer_seq[t])) for t in range(self.context.buffer_size)],
            dtype=np.int8,
        )

    @property
    def active_daemons(self) -> np.ndarray[tuple[int, ...], np.dtype[np.bool]]:
        """
        Array with each filed indicating whether each daemon sequence is active.

        Formatted for directly creating ``Solution`` instance.
        :return: 1d binary with indicator for each daemon sequence.
        """
        self._require_finished()
        return np.array([round(self.context.model.getVal(var)) for var in self.context.y], dtype=np.bool)

    @property
    def total_points(self) -> np.int64:
        """
        Amount of earned points

        Formatted for directly creating ``Solution`` instance.
        :return: sum of points for each activated daemon.
        """
        self._require_finished()
        return np.int64(np.dot(self.context.daemons_costs, self.active_daemons))
