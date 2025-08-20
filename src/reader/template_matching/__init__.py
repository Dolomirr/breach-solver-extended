# src/image_reader/template_matching/__init__.py
"""
High-level image-to-task scanner, uses cv2 template matching.

.. seealso::
:class:`reader.ScannerTemplates`

Examples:
    >>> from reader import ScannerTemplates, image_loader
    >>> path = pathlib.Path("/path/to/image_to_scan.png")
    >>> image = image_loader.from_path(path)
    >>> scanner = ScannerTemplates()
    >>> soft_task = scanner.read(image)

"""

from .scanner import ScannerTemplates
from .structs import TemplateProcessingConfig

__all__ = [
    "ScannerTemplates",
    "TemplateProcessingConfig",
]
