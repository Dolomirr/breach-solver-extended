import logging
from typing import Self, cast

import cv2
import numpy as np

from core import setup_logging

from ..image_loader import ColoredImage, GrayScaleImage
from .structs import Images, TemplateProcessingConfig

setup_logging()
log = logging.getLogger(__name__)


class ImageProcessor:
    """
    Responsible for setting stages of images in ``reader.template_matching.structs.Images``
    
    Due to how template matching works, some stages are dependent on previous ones, and directly derived from previous version.
    ``Images``
    
    Methods:
        set_base: provide ``Images`` instance with base colored image.
        set_resized: pads and resize base image to fit in best ration for template matching.
        set_grayed: converts colored image to grayscale.
        set_binary: converts grayscale image binarized image in {0, 1} values range.
        set_buffer: cut out region with buffer cells.
        set_buffer_binary: converts buffer region binarized image in {0, 1} values range using dynamic threshold.
    
    :param config:
    :type config: TemplateProcessingConfig

    """
    
    config: TemplateProcessingConfig
    images: Images

    def __init__(self, config: TemplateProcessingConfig):
        self.config = config

    def _split_in(self, img: GrayScaleImage, pos, axis=0) -> tuple[GrayScaleImage, GrayScaleImage]:
        """
        Splits provided as 2d numpy array (gray cmap) into two separate in given position and axis.

        :param img: image
        :param pos: position x/y
        :param axis: 0 - horizontal 1 - vertical, defaults to 0
        :returns: tuple(top, bottom) or tuple(left, right)
        """
        res = (
            (img[:pos, :], img[pos:, :])
            if axis == 0
            else (img[:, :pos], img[:, pos:])
            )  # fmt: skip

        return cast("tuple[GrayScaleImage, GrayScaleImage]", res)

    def set_base(self, images: Images) -> Self:
        """
        Set base processed image.
        """
        self.images = images
        return self

    def set_resized(self) -> Self:
        """
        Set ``Image.sized`` attribute.
        
        Resize and pad image to fit within a target size while hard maintaining aspect ratio, needed for proper template matching.

        Uses:
            :attr:`config.TARGET_SIZE`
        
        """
        target_width, target_height = self.config.TARGET_SIZE
        h, w = self.images.raw.shape[:2]

        scale = min(target_width / w, target_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(self.images.raw, (new_w, new_h), interpolation=interpolation)

        padded = np.zeros((target_height, target_width, self.images.raw.shape[2]), dtype=self.images.raw.dtype)

        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2

        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        self.images.sized = cast('ColoredImage', padded)    # safe, because images.raw is guarantied to be 3-layered
        log.debug("Resized set.", extra={"target_size": self.config.TARGET_SIZE})
        return self

    def set_grayed(self) -> Self:
        """
        Sets the ``Image.gray`` attribute.
        
        Convert 3 channel colored image to gray-scale one channel.
        """
        # safe, due to dtype check in image_loader.ImageReader
        self.images.gray = cast("GrayScaleImage", cv2.cvtColor(self.images.sized, cv2.COLOR_BGR2GRAY))
        log.debug("Grayed set.")
        return self

    def set_binary(self) -> Self:
        """
        Sets the ``Image.binary`` attribute.
        
        Binarize grayscale image to {0, 1} using global thresholding.

        Uses:
            :attr:`config.MINVAL_THRESHOLD`
            :attr:`config.MAXVAL_THRESHOLD`
        """
        val, img_binary = cv2.threshold(
            self.images.gray,
            self.config.MINVAL_THRESHOLD,
            self.config.MAXVAL_THRESHOLD,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        # safe, due to dtype check in image_loader.ImageReader
        self.images.binary = cast("GrayScaleImage", img_binary)
        log.debug("Binary set.", extra={"val": val})
        return self

    def set_buffer(self, vert_bound, hor_bound) -> Self:
        """
        Sets the ``Image.buffer_cut`` attribute.
        
        Based on provided bounds cut region containing buffer.
        """
        _, right = self._split_in(self.images.gray, vert_bound, axis=1)
        upper, _ = self._split_in(self.images.gray, hor_bound, axis=0)

        self.images.buffer_cut = upper
        log.debug("Buffer cutter.", extra={"hor_bound": hor_bound, "vert_bound": vert_bound})
        return self

    def set_buffer_binary(self) -> Self:
        """
        Sets the ``Image.buffer_binary`` attribute.
        
        Binarize grayscale image containing buffer cells to {0, 1} using dynamic thresholding.
        """
        buffer_img_shape = min(self.images.buffer_cut.shape)
        thres_block_size = (
            buffer_img_shape
            if buffer_img_shape % 2 != 0
            else buffer_img_shape - 1
            )  # fmt: skip

        buffer_thresh = cv2.adaptiveThreshold(
            self.images.buffer_cut,
            maxValue=self.config.MAXVAL_THRESHOLD,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=thres_block_size,
            C=-10,
        )

        self.images.buffer_binary = cast("GrayScaleImage", buffer_thresh)
        log.debug("Buffer binary set.", extra={"block_size": thres_block_size})
        return self
