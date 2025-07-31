from dataclasses import dataclass
from typing import Self

from numpy import array_equal, dtype, int8, issubdtype, ndarray

# seems like pyright does not fully support new numpy's annotation
# so i cant specify dimensions :/
type ArrayInt8 = ndarray[tuple[int, ...], dtype[int8]]


@dataclass(frozen=True, slots=True)
class Task:
    """
    Represents single valid task for breach protocol.

    :param matrix: 2d ndarray of dtype int8
    :param daemons: 2d ndarray of dtype int8
    :param daemons_costs: 1d ndarray of dtype int8
    :param buffer_size: int8
    """

    # seems like pyright does not fully support new numpy's annotation
    matrix: ArrayInt8
    daemons: ArrayInt8
    daemons_costs: ArrayInt8
    buffer_size: int8

    def __post_init__(self) -> None:
        msg = []
        if not issubdtype(self.matrix.dtype, int8):
            msg.append(f"matrix must be of dtype numpy.int8, given: {self.matrix.dtype}")
        if self.matrix.ndim != 2:
            msg.append("Matrix must be 2d")
        if self.daemons.ndim != 2:
            msg.append("daemons must be 2d ndarray")
        if self.daemons.shape[0] != self.daemons_costs.shape[0]:
            msg.append(
                f"demons and daemons_costs must have the same length, given: {self.daemons.shape[0]}, {self.daemons_costs.shape[0]}",
            )
        if not (issubdtype(self.buffer_size, int8) and self.buffer_size > 0):
            msg.append(
                f"buffer size must be int8 and non-negative, given: {type(self.buffer_size)}, {self.buffer_size}",
            )
        if msg:
            msg = "\n" + "\n".join(msg)
            raise ValueError(msg)
    
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return (
            array_equal(self.matrix, other.matrix)
            and array_equal(self.daemons, other.daemons)
            and array_equal(self.daemons_costs, other.daemons_costs)
            and self.buffer_size == other.buffer_size
        )
