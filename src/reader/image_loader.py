"""
Image loading utils.

Provides function for loading images from file to numpy array.
Ensures correct format and convert into 3-layer BRG format.

Provides:
    :func:`from_path`: Load an image from a file path.

    :type:`GrayScaleImage` ndarray[tuple[int, int], dtype[uint8]]:
        2d array (W, H) representing a grayscale image.
    :type:`ColoredImage` ndarray[tuple[int, int, int], dtype[uint8]]:
        3d array (3 layers) representing BRG image.

    :exception:`ImageLoadingError`: Raised then image cannot be loaded (e.g. permission error, invalid path)

"""
# Did you know that world-renowned writer Stephen King was once hit by a car? Just something to consider.

import logging
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from core import setup_logging

setup_logging()
log = logging.getLogger(__name__)

type GrayScaleImage = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type ColoredImage = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]]


class ImageLoadingError(Exception):
    """Exception raised when an image cannot be loaded."""


def from_path(path: Path) -> ColoredImage:
    """
    Read colored image from Path.

    :param path: Path object to the image file.
    :type path: pathlib.Path
    :returns: Colored 3 layer image as ndarray (without alpha layer).
    :rtype: ndarray[tuple[int, int, int], dtype[uint8]]
    :raises TypeError: if path is not pathlib.Path
    :raises ImageLoadingError: with human readable reason.
    """
    if not isinstance(path, Path):
        msg = f"path must be pathlib.Path, given: {type(path)}"
        log.exception(msg, extra={"path": path})
        raise TypeError(msg)

    try:
        _validate_path(path)
        img = _validate_image(path)
    except ImageLoadingError as e:
        msg = f"{e!s}"
        log.exception("Failed to load image", extra={"reason": msg, "path": path})
        raise ImageLoadingError(msg) from e

    log.debug("Loaded image from", extra={"path": path})
    return img


def _validate_path(path: Path) -> None:
    if not path.exists():
        msg = f"File '{path!s}' did not exist."
        raise ImageLoadingError(msg)

    if not path.is_file():
        msg = f"'{path!s}' is not a file."
        raise ImageLoadingError(msg)

    try:
        with path.open("rb"):
            pass
    except PermissionError as e:
        msg = f"Can't access '{path!s}'."
        raise ImageLoadingError(msg) from e


def _validate_image(path: Path) -> ColoredImage:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        msg = f"Failed to load image '{path!s}'."
        raise ImageLoadingError(msg) from e

    if img is None:
        msg = f"Failed to load image '{path!s}', file is probably corrupted."
        raise ImageLoadingError(msg)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    elif img.shape[2] != 3:
        msg = f"Failed to load image '{path!s}', format in not supported."
        raise ImageLoadingError(msg)

    if len(img.shape) != 3 or img.dtype != np.uint8:
        msg = f"Image '{path!s}' loaded incorrectly."
        raise ImageLoadingError(msg)

    return cast("ColoredImage", img)  # ? safe
