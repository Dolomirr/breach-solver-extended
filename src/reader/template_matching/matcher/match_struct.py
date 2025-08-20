from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Self, final

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


@dataclass(frozen=True, slots=True)
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
    :type bbox: BBox[int, int, int, int] (NamedTuple)
    :type center: Center[int, int] (NamedTuple)
    """

    label: str
    template_idx: int
    score: float
    bbox: BBox
    """
    NamedTuple:
        x1: int
        y1: int
        x2: int
        y2: int
    """
    center: Center
    """
    NamedTuple:
        cx: int
        cy: int
    """

    def __str__(self) -> str:
        return f"Match({self.label}, ({self.center.cx}, {self.center.cy})"


# This could be done with metaclass but null match really only needed for single method (MatchGrouper.structure_matrix),
# to make it aware of undetected symbols in grid, so metaclass would be a little overkill
@final  # just to be sure
@dataclass(frozen=True, slots=True)
class NullMatch(Match):
    """
    Missing version of ``Match``.
    
    Value for label taken directly from ``HEX_DISPLAY_MAP`` for easier mapping to actual visible strings later in ui.

    Rest of attributes corresponds to attributes of ``Match`` and contain placeholder values.
    Support direct instance check with ``is NullMatch``.
    
    .. important::
        To avoid possible re-calculation of position for bbox, and centers all coordinates set to `-1`,
        and therefore it is required to explicitly ignore negative values in
        all operations involving some calculations based on them.
        (which is anyway not possible since we are dealing with array-based image representation).
    
    .. important::
        Instance should not be created directly, instead use ``.getin`` method.

    
    :param label: Matched template's label.
        Default: HEX_DISPLAY_MAP[HexSymbol.S_BLANK] (" ▧").
    :param template_idx: Index of the template image for corresponding label.
        Default: -1.
    :param score: Confidence score of the match.
        Default: -1.0
    :param bbox: Bounding box of the matched region: (x1, y1, x2, y2).
        Default: BBox(-1, -1, -1, -1)
    :param center: Center of the bbox: (cx, cy).
        Default: Center(-1, -1).

    :type label: str
    :type template_idx: int
    :type score: float
    :type bbox: BBox[int, int, int, int] (NamedTuple)
    :type center: Center[int, int] (NamedTuple)

    """

    label: str = HEX_DISPLAY_MAP[HexSymbol.S_BLANK]
    """HEX_DISPLAY_MAP[HexSymbol.S_BLANK] (" ▧")"""
    template_idx: int = -1
    """-1"""
    score: float = -1.0
    """-1.0"""
    bbox: BBox = BBox(-1, -1, -1, -1)  # noqa: RUF009 not an issue, designed as lazy singleton
    """BBox(-1, -1, -1, -1)"""
    center: Center = Center(-1, -1)  # noqa: RUF009
    """Center(-1, -1)"""

    @classmethod
    @lru_cache(maxsize=1)
    def instance(cls) -> Self:
        """
        Always return same instance of NullMatch.

        :rtype: NullMatch
        """
        return cls()

    def __str__(self) -> str:
        return "NullMatch()"

    __repr__ = __str__
