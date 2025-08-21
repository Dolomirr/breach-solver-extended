import logging
from dataclasses import dataclass
from typing import Self

import numpy as np

from core import mapper_to_int, setup_logging
from core.base_setup import HEX_DISPLAY_MAP, HexSymbol

from .task import Task

setup_logging()
log = logging.getLogger(__name__)


@dataclass
class SoftTask:
    """
    Temporary mutable data structure for representing task.

    Any manipulations with internal data structures should not be done directly.

    :param matrix: matrix
    :type matrix: list[list[str]]
    :param daemons: daemons sequence
    :type daemons: list[list[str]]
    :param buffer_size: buffer size
    :type buffer_size: int
    :param costs: Optional list of costs of each daemon.
        Default: If not provided, it will be calculated as double length of each sequence.
    :type costs: list[int]

    :raises ValueError: If any of the input parameters are invalid.
    """

    _matrix: list[list[str]]
    _daemons: list[list[str]]
    _buffer_size: int
    _costs: list[int]

    def __init__(
        self,
        matrix: list[list[str]],
        daemons: list[list[str]],
        buffer_size: int,
        costs: list[int] | None = None,
    ) -> None:
        msgs: list[str] = []
        if not isinstance(matrix, list):
            msgs.append(f"matrix must be list, given: {type(matrix)}")
        if any(not isinstance(row, list) for row in matrix):
            msgs.append("each row in matrix must be a list")
        if any(None in row for row in matrix):
            msgs.append("matrix cannot contain None values")

        if not isinstance(daemons, list):
            msgs.append(f"daemons must be list, given: {type(daemons)}")
        if any(not isinstance(row, list) for row in daemons):
            msgs.append("each row in daemons must be a list")
        if any(None in row for row in daemons):
            msgs.append("daemons cannot contain None values")
        if not (isinstance(buffer_size, int) and buffer_size >= 0):
            msgs.append(f"buffer_size must be a non-negative integer, given: {buffer_size}")

        if costs is not None:
            if not isinstance(costs, list):
                msgs.append(f"costs must be list, given: {type(costs)}")
            if any(not isinstance(cell, int) for cell in costs):
                msgs.append("each row in costs must contain integers")
        else:
            # updated in recalc_costs
            costs = []

        if msgs:
            msg = "\n" + "\n".join(msgs)
            log.exception(msg)
            raise ValueError(msg)

        self._matrix = matrix
        self._daemons = daemons
        self._buffer_size = buffer_size
        self._costs = costs
        self.recalc_costs()

    def recalc_costs(self) -> Self:
        """
        Calculates costs of each sequence of daemons from zero, overrides all previously set costs.
        """
        self._costs = [len(row) for row in self._daemons]
        return self

    @property
    def _padded_daemons(self) -> list[list[str]]:
        max_len = max(len(row) for row in self._daemons)
        pad_str = HEX_DISPLAY_MAP[HexSymbol.S_STOP]
        daemons = []
        for sequence in self._daemons:
            row = sequence.copy()
            cur_len = len(row)
            if cur_len < max_len:
                row.extend([pad_str] * (max_len - cur_len))
            daemons.append(row)
        return daemons

    def make_hard(self) -> Task:  # 0_o
        """
        Converts self to frozen ``Task``.
        """
        try:
            task = Task(
                matrix=np.array(
                    [[mapper_to_int(cell) for cell in row] for row in self._matrix],
                    dtype=np.int8,
                ),
                daemons=np.array(
                    [[mapper_to_int(cell) for cell in row] for row in self._padded_daemons],
                    dtype=np.int8,
                ),
                daemons_costs=np.array(self._costs, dtype=np.int8),
                buffer_size=np.int8(self._buffer_size),
            )
        except Exception as e:
            msg = f"Error converting SoftTask to Task: {e}"
            log.exception(msg, extra={"current_state": self})
            raise ValueError(msg) from e

        return task
