from dataclasses import dataclass
from typing import Self

import numpy as np

# seems like pyright does not fully support new numpy's annotation
# so i cant specify dimensions :/
type ArrayInt8 = np.ndarray[tuple[int, ...], np.dtype[np.int8]]


@dataclass(frozen=True, slots=True)
class Task:
    """
    Represents single valid task for breach protocol.
    
    Support hashing and equality checks.
    
    Methods
    -------
        copy
            Shortcut to __copy__().
        is_identical
            Comparison with other Task.

    :param matrix: rectangular 2d np.ndarray, each element represents hex symbol, according to ``HexSymbol``.
    :type matrix: np.ndarray[tuple[int, int8], np.dtype[np.int8]]
    :param daemons: rectangular 2d np.ndarray, each element represents daemon symbol, according to ``HexSymbol``,
        for shorter daemons sequences are padded with ``HexSymbol.S_STOP`` == -1.
    :type daemons: np.ndarray[tuple[int, int8], np.dtype[np.int8]]
    :param daemons_costs: 1d np.ndarray each represents cost in points of corresponding daemon.
    :type daemons_costs: np.ndarray[tuple[int], np.dtype[np.int8]]
    :param buffer_size: np.int8
    :type buffer_size: np.int8

    """

    matrix: ArrayInt8
    daemons: ArrayInt8
    daemons_costs: ArrayInt8
    buffer_size: np.int8

    def __post_init__(self) -> None:
        msg: list[str] = []
        if not np.issubdtype(self.matrix.dtype, np.int8):
            msg.append(f"matrix must be of np.dtype numpy.np.int8, given: {self.matrix.dtype}")
        if self.matrix.ndim != 2:
            msg.append("Matrix must be 2d")
        if self.daemons.ndim != 2:
            msg.append("daemons must be 2d np.ndarray")
        if self.daemons.shape[0] != self.daemons_costs.shape[0]:
            msg.append(
                f"demons and daemons_costs must have the same length, given: {self.daemons.shape[0]}, {self.daemons_costs.shape[0]}",
            )
        if not (np.issubdtype(self.buffer_size, np.int8) and self.buffer_size > 0):
            msg.append(
                f"buffer size must be np.int8 and non-negative, given: {type(self.buffer_size)}, {self.buffer_size}",
            )
        if msg:
            msgs = "\n" + "\n".join(msg)
            raise ValueError(msgs)

    def __copy__(self) -> Self:
        cls = type(self)
        return cls(
            matrix=self.matrix.copy(),
            daemons=self.daemons.copy(),
            daemons_costs=self.daemons_costs.copy(),
            buffer_size=self.buffer_size,
        )

    def copy(self) -> Self:
        return self.__copy__()

    def __hash__(self) -> int:
        return hash(
            (
                self.matrix.tobytes(),
                self.daemons.tobytes(),
                self.daemons_costs.tobytes(),
                int(self.buffer_size),
            ),
        )

    def is_identical(self, other: object) -> bool:
        """
        Checks if the current object is identical to another ``Task`` object.
        
        :return: True if tasks are identical, False otherwise, NotImplemented if ``other`` is not a ``Task`` object.
        """
        if not isinstance(other, Task):
            return NotImplemented
        return (
            np.array_equal(self.matrix, other.matrix)
            and np.array_equal(self.daemons, other.daemons)
            and np.array_equal(self.daemons_costs, other.daemons_costs)
            and self.buffer_size == other.buffer_size
        )
