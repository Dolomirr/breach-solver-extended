from dataclasses import dataclass
from functools import total_ordering
from typing import Self

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .task import Task

type ArrayInt8 = np.ndarray[tuple[int, ...], np.dtype[np.int8]]
type ArrayBool = np.ndarray[tuple[int, ...], np.dtype[np.bool]]
type SolverResult = Solution | NoSolution


@total_ordering
@dataclass(frozen=True, slots=True)
class Solution:
    """
    Represents single valid solution for breach protocol.
    Supports full comparison and ordering by ``total_points`` set with other solutions by total points.
    Suppers hashing.
    
    Methods
    -------
        is_identical
            Comparison by all attributes.
        from_task
            Allow to reconstruct full solution from only path, and corresponding ``Task``.

    :param path: 2d array with shape (n, 2), represents sequentially picked cells on matrix, each pair is coordinates of cell.
    :type path: np.ndarray[tuple[int, int], np.dtype[np.int8]]
    :param buffer_sequence: 1d array, represents symbols picked from matrix, corresponds to path, therefore length must match the length of ``path``.
    :type buffer_sequence: np.ndarray[tuple[int], np.dtype[np.int8]]
    :param active_daemons: 1d array, each value indicates whether corresponding daemon active.
    :type active_daemons: np.ndarray[tuple[int], np.dtype[np.bool]]
    :param total_points: Sum of pits earned by activating daemons, must be positive.
    :type total_points: np.int64

    """

    path: ArrayInt8
    buffer_sequence: ArrayInt8
    active_daemons: ArrayBool
    total_points: np.int64  # left signed to avoid casting in cpp modules

    def __post__init__(self) -> None:
        msg: list[str] = []
        if self.path.ndim != 2 or self.path.shape[0] == 0 or self.path.shape[1] != 2:
            msg.append(f"Path must be a 2D array with shape (n, 2), and n>0, given: {self.path.shape}")
        if self.buffer_sequence.ndim != 1 or self.buffer_sequence.shape[0] != self.path.shape[0]:
            msg.append(f"buffer_sequence must be 1d and match path length, given: {self.buffer_sequence.shape}")
        if self.active_daemons.ndim != 1:
            msg.append(f"active_daemons must be 1d, given: {self.active_daemons.ndim}")
        if not (np.issubdtype(self.total_points, np.int64) and self.total_points > 0):
            msg.append(f"Total points must be a positive integer, given: {self.total_points}")
        if msg:
            msgs = "\n" + "\n".join(msg)
            raise ValueError(msgs)

    # TODO! widen down Exception
    # TODO!: change names, add verification for already existing paths?
    @classmethod
    def from_task(cls, path: ArrayInt8, task: Task) -> SolverResult:
        """
        Creates instance of ``SolverResult`` from path (minimal needed information to reconstruct a solution) and valid ``Task`` instance fields.
        
        :param path: Must be valid for norma Solution constructor.
        :param task: ``Task`` instance.
        :return: ``Solution`` or ``NoSolution`` if no solution for given path and Task exist.
        :raises: ``ValueError`` if provided path have incorrect shape or type.
        """
        msg: list[str] = []
        if not isinstance(path, np.ndarray):
            msg.append(f"Path must be a numpy array, given: {type(path)}\n")
        else:
            if path.ndim != 2:
                msg.append(f"Path must be a 2d array, given: {path.ndim}")
            if path.shape[1] != 2:
                msg.append(f"path must have shape (n, 2), given: {path.shape}")
        if msg:
            msgs = "\n" + "\n".join(msg)
            raise ValueError(msgs)
        
        if path.shape[0] == 0:
            return NoSolution("Received empty path.")

        try:
            buffer_size = task.buffer_size
            buffer = np.zeros(buffer_size, dtype=np.int8)
            buffer[: path.shape[0]] = task.matrix[path[:, 0], path[:, 1]]
            buffer_sequence = buffer[: path.shape[0]]
        except (IndexError, TypeError) as e:
            return NoSolution(reason=f"Failed to construct buffer_sequence: \n{e!r}")

        try:
            daemons = task.daemons
            num_daemons = len(daemons)
            active_daemons = np.zeros(num_daemons, dtype=bool)

            for i in range(num_daemons):
                d = daemons[i]
                d_len = d.shape[0]
                if d_len > buffer_sequence.shape[0]:
                    continue
                windows = sliding_window_view(buffer_sequence, window_shape=d_len)
                if (windows == d).all(axis=1).any():
                    active_daemons[i] = True
        except (ValueError, TypeError, IndexError) as e:
            return NoSolution(f"Failed to construct active_demons: \n{e!r}")

        try:
            total_points = np.int64(task.daemons_costs @ active_daemons)
        except (ValueError, TypeError) as e:
            return NoSolution(f"Failed to compute total_points: \n{e!r}")
        
        try:
            return cls(
                path=path,
                buffer_sequence=buffer_sequence,
                active_daemons=active_daemons,
                total_points=total_points,
            )
        except (ValueError, TypeError) as e:
            return NoSolution(f"Error while construction Solution: \n{e!r}")

    def __copy__(self) -> Self:
        cls = type(self)
        return cls(
            path=self.path.copy(),
            buffer_sequence=self.buffer_sequence.copy(),
            active_daemons=self.active_daemons.copy(),
            total_points=self.total_points,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.path.tobytes(),
                self.buffer_sequence.tobytes(),
                self.active_daemons.tobytes(),
                int(self.total_points),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        return self.total_points == other.total_points

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        if self.total_points != other.total_points:
            return self.total_points != other.total_points
        return self.path.shape[0] < other.path.shape[0]

    def is_identical(self, other: object) -> bool:
        """
        Checks if the current object is identical to another ``Task`` object.
        
        :return: True if tasks are identical, False otherwise, NotImplemented if ``other`` is not a ``Solution`` object.
        """
        if not isinstance(other, Solution):
            return NotImplemented
        return (
            np.array_equal(self.path, other.path)
            and np.array_equal(self.buffer_sequence, other.buffer_sequence)
            and np.array_equal(self.active_daemons, other.active_daemons)
            and self.total_points == other.total_points
        )


@dataclass
class NoSolution:
    """
    Indicates that no valid solution exist or could bew found.
    
    :param reason: The reason for the absence of a solution.
    :type reason: str
    """

    reason: str
