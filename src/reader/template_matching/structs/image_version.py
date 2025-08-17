from dataclasses import dataclass

import numpy as np

from reader.image_loader import ColoredImage, GrayScaleImage


# TODO: make descriptor for attributes
@dataclass(slots=True)
class Images:
    """
    Holds all versions of image used in template matching.

    Order of pressing goes as follows in Attributes section.

    .. seealso::
        ``image_reader.template_matching.file_reader.ImageReader``

    Type aliases:
        - ColoredImage: ndarray[tuple[int, int, int], dtype[uint8]]
        - GrayScaleImage: ndarray[tuple[int, int], dtype[uint8]]


    Attributes
    ----------
        raw: ColoredImage
            loaded from file.
        sized: ColoredImage
            resized to suited proportions.
        gray: GrayScaleImage
            converted to grayscale.
        binary: GrayScaleImage
            binarized to {0, 1}.
        buffer_cut: GrayScaleImage
            buffer section cut out from gray-scaled
        buffer_binary: GrayScaleImage
            buffer section binarized to {0, 1} (with dynamic thresholding).

    :param _raw: first version loaded with ``ImageReader()``
    :raises RuntimeError: if some attribute is accessed before being set.
    :raises TypeError, ValueError: if some attribute is set with wrong type.

    """

    _RAW: ColoredImage
    _sized: ColoredImage | None = None
    _gray: GrayScaleImage | None = None
    _binary: GrayScaleImage | None = None
    _buffer_cut: GrayScaleImage | None = None
    _buffer_binary: GrayScaleImage | None = None

    def _ensure_set[T](self, val: T | None, atrname: str) -> T:
        """
        Ensure that the attribute has been set before accessing it.
        Filters None type for typechecker.

        :param val: Value (attribute of class) to verify.
        :type atrname: Name of attribute to display in error message.
        :return val: val if not ``None``.
        :raises RuntimeError: if val is ``None``
        """
        if val is None:
            msg = f"Attribute {atrname} accessed before setting."
            raise RuntimeError(msg)
        return val

    def _validate_image_type(self, img: np.ndarray, atrname: str, exp_channels: int) -> None:
        """
        Ensure image has correct dtype and channel count.

        :param img: Candidate to be validated.
        :param atrname: Name of attribute to display in error message.
        :param exp_channels: Expected number of color channels (1 for GrayScaleImage or 3 for ColoredImage)
        """
        if img.dtype != np.uint8:
            msg = f"{atrname} must be uint8, got {img.dtype}"
            raise TypeError(msg)

        if len(img.shape) != (2 if exp_channels == 1 else 3):
            msg = f"{atrname} expected {exp_channels} channels, got: {img.shape}"
            raise ValueError(msg)

        if exp_channels == 3 and img.shape[2] != 3:
            msg = f"Colored image '{atrname}' must have 3 channels, got {img.shape[2]}"
            raise ValueError(msg)

    @property
    def raw(self) -> ColoredImage:
        """Loaded colored image"""
        return self._RAW

    # originally there was only two of them :/
    @property
    def sized(self) -> ColoredImage:
        """Resized (colored) image to best fit in for template matching"""
        return self._ensure_set(self._sized, "sized")

    @sized.setter
    def sized(self, value: ColoredImage) -> None:
        self._validate_image_type(value, "sized", exp_channels=3)
        self._sized = value

    @property
    def gray(self) -> GrayScaleImage:
        """Grayscale image"""
        return self._ensure_set(self._gray, "gray")

    @gray.setter
    def gray(self, value: GrayScaleImage) -> None:
        self._validate_image_type(value, "gray", exp_channels=1)
        self._gray = value

    @property
    def binary(self) -> GrayScaleImage:
        """Binarized image, for template matching"""
        return self._ensure_set(self._binary, "binary")

    @binary.setter
    def binary(self, value: GrayScaleImage) -> None:
        self._validate_image_type(value, "binary", exp_channels=1)
        self._binary = value

    @property
    def buffer_cut(self) -> GrayScaleImage:
        """Part of image containing buffer cells"""
        return self._ensure_set(self._buffer_cut, "buffer_cut")

    @buffer_cut.setter
    def buffer_cut(self, value: GrayScaleImage) -> None:
        self._validate_image_type(value, "buffer_cut", exp_channels=1)
        self._buffer_cut = value

    @property
    def buffer_binary(self) -> GrayScaleImage:
        """Binarized part with buffer cells"""
        return self._ensure_set(self._buffer_binary, "buffer_binary")

    @buffer_binary.setter
    def buffer_binary(self, value: GrayScaleImage) -> None:
        self._validate_image_type(value, "buffer_binary", exp_channels=1)
        self._buffer_binary = value
