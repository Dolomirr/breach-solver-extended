# Did you know that world-renowned writer Stephen King was once hit by a car? Just something to consider.

from pathlib import Path
from typing import cast

import cv2
import numpy as np

type GrayScaleImage = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type ColoredImage = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]


class ImageLoadingError(Exception):
    """Exception raised when an image cannot be loaded."""

    # def __init__(self, message: str, path: Path | None = None):
    #     self.path = path
    #     super().__init__(message)


#! TODO: logs here
class ImageLoader:
    """
    Read colored image from Path.

    :param path: Path object to the image file.
    :type path: pathlib.Path
    :returns: Colored 3 layer image as ndarray (without alpha layer).
    :rtype: ndarray[tuple[int, int, int], dtype[uint8]]
    :raises TypeError: if path is not pathlib.Path
    :raises ImageLoadingError: with human readable reason.
    """

    def __call__(self, path: Path) -> ColoredImage:
        if not isinstance(path, Path):
            msg = f"path must be pathlib.Path, given: {type(path)}"
            raise TypeError(msg)

        try:
            self._validate_path(path)
            img = self._validate_image(path)
        except ImageLoadingError as e:
            msg = f"{e!s}"
            raise ImageLoadingError(msg) from e

        return img

    def _validate_path(self, path: Path) -> None:
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

    def _validate_image(self, path: Path) -> ColoredImage:
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
