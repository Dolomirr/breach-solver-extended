from typing import Protocol

from core import SoftTask

from .image_loader import ColoredImage


class ImageReader[ReaderConfig](Protocol):
    def __init__(self, config: ReaderConfig | None = None) -> None: ...

    def read(self, image: ColoredImage) -> SoftTask: ...
