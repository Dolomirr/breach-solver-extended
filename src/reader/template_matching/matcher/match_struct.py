from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Self

from core import HEX_DISPLAY_MAP, HexSymbol


class BBox(NamedTuple):
    """
    ``Match.bbox``

    Coordinates of bounding box on image.
        - (x1, y1) top-left corner
        - (x2, y2) bottom-right corner

    :param x1:
    :type x1: int
    :param x2:
    :type x2: int
    :param y1:
    :type y1: int
    :param y2:
    :type y2: int
    """

    x1: int
    y1: int
    x2: int
    y2: int


class Center(NamedTuple):
    """
    ``Match.center``

    Coordinates center of match.
        - (cx, cy)

    :param cx:
    :type cx: int
    :param cy:
    :type cy: int
    """

    cx: int
    cy: int


@dataclass(frozen=True)
class Match:
    """
    Single template match found within a larger image.

    :param label: Matched template's label.
    :param template_idx: Index of the template image for corresponding label.
    :param score: Confidence score of the match.
    :param bbox: Bounding box of the matched region: (x1, y1, x2, y2).
    :param center: Center of the bbox: (cx, cy).

    :type label: str
    :type template_idx: int
    :type score: float
    :type bbox: BBox[int, int, int, int]
    :type center: Center[int, int]
    """

    label: str
    template_idx: int
    score: float
    bbox: BBox
    center: Center

    def __str__(self) -> str:
        return f"Match({self.label}, ({self.center.cx}, {self.center.cy})"


# this could be done with metaclass but null match really only needed for structuring in matrix,
# to make it aware of undetected symbols in grid
@dataclass(frozen=True)
class NullMatch(Match):
    label: str = HEX_DISPLAY_MAP[HexSymbol.S_BLANK]
    template_idx: int = -1
    score: float = -1.0
    bbox: BBox = BBox(-1, -1, -1, -1)  # noqa: RUF009 not an issue, designed as lazy singleton
    center: Center = Center(-1, -1)  # noqa: RUF009

    _instances: Self | None = None

    @classmethod
    def getin(cls) -> Self:
        if cls._instances is None:
            cls._instances = cls()
        return cls._instances

    def __str__(self) -> str:
        return "NullMatch()"

    def __repr__(self) -> str:
        return "NullMatch()"
