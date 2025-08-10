from dataclasses import dataclass
from typing import NamedTuple


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
        return f"Match({self.label}, ({self.center.cx}, {self.center.cy}))"
