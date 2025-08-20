# src/image_reader/__init__.py
"""
Contains classes that allow to read image from different sources and ways.

Provides:
    :module:`image_loader`: Contains functions for loading image from file
    :class:`ScannerTemplates`: Scans image using template matching.

"""


from .image_loader import ImageLoadingError, from_path
from .template_matching import ScannerTemplates, TemplateProcessingConfig

__all__ = [
    "ImageLoadingError",
    "ScannerTemplates",
    "TemplateProcessingConfig",
    "from_path",
]
