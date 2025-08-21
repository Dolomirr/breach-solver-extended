from dataclasses import dataclass


# configs placed INSIDE corresponding modules, because:
# 1. im not sure yet what ways to input will be supported.
# 2. they not suppose to be changed on runtime nor before start of scanning.
@dataclass(frozen=True, slots=True)
class TemplateProcessingConfig:
    """
    A class to hold configuration settings for template processing.
    
    .. Important:
        Does not meant to be changed during runtime, may cause errors during templates loading or matches post-processing.

    Attributes:
        TARGET_SIZE (tuple[int, int]): The target size for images.
            Default: (2048, 1080)
        MINVAL_THRESHOLD (int): The lower value threshold used in binarization by ``ImageProcessor.set_binary``.
            Default: 0
        MAXVAL_THRESHOLD (int): The higher value threshold used in binarization by ``ImageProcessor.set_binary`` and ``ImageProcessor.set_buffer_binary``.
            Default: 1
        EXISTING_TEMPLATES (frozenset[str]): A set of existing template names used in ``TemplateLoader.load``.
            Default: frozenset({'1C', '55', 'BD', 'E9', '7A', 'FF', 'X9', 'XX', 'XH', 'IX', 'XR'})
        BUFFER_TEMPLATES (str): The buffer template name used in ``TemplateLoader.load``.
            Default: "BUFFER_CELL"
        ADDITIONAL_TEMPLATES (frozenset[str]): A set of additional template names that are not currently used but might be used later.
            Default: frozenset()
        MATCHING_THRESHOLD (float): The minimal matching threshold used in ``TemplateMatcher.match``.
            Default: 0.65
        OVERLAP_THRESHOLD (float): The overlap threshold used in ``TemplateMatcher.match``.
            Default: 0.01
        CLUSTERING_EPS (float | None): The epsilon value for clustering used in ``MatchGrouper.filter_unclustered``.
            Default: None
        CLUSTERING_EPS_FACTOR (float): A factor to calculate the epsilon value for clustering used in ``MatchGrouper.filter_unclustered``.
            Default: 2.8284 (approximately 2 * sqrt(2))
        CLUSTERING_MIN_SAMPLES (int): The minimum number of samples required for clustering used in ``MatchGrouper.filter_unclustered``.
            Default: 3

    """

    TARGET_SIZE: tuple[int, int] = (2048, 1080)
    """Used in ``ImageProcessor.set_resized``."""
    MINVAL_THRESHOLD: int = 0
    """Used in ``ImageProcessor.set_binary`` as lower value in binarization."""
    MAXVAL_THRESHOLD: int = 1
    """Used in ``ImageProcessor.set_binary`` and ``ImageProcessor.set_buffer_binary`` as higher value in binarization"""

    # This is separate from core.base_setup because it specific to template matching and should not be changed on runtime.
    EXISTING_TEMPLATES: frozenset[str] = frozenset(
        (
            "1C",  # base game
            "55",
            "BD",
            "E9",
            "7A",
            "FF",
            "X9",  # dlc
            "XX",
            "XH",
            "IX",
            "XR",
        ),
    )
    """Used in ``TemplateLoader.load``."""
    BUFFER_TEMPLATES: str = "BUFFER_CELL"
    """Used in ``TemplateLoader.load``."""
    ADDITIONAL_TEMPLATES: frozenset[str] = frozenset()  # not used currently, maybe something later
    """Used in ``TemplateLoader.load``."""
    MATCHING_THRESHOLD: float = 0.65
    """Used in ``TemplateMatcher.match`` as minimal """
    OVERLAP_THRESHOLD: float = 0.01
    """Used in ``TemplateMatcher.match``."""
    CLUSTERING_EPS: float | None = None
    """Used in ``MatchGrouper.filter_unclustered``."""
    CLUSTERING_EPS_FACTOR: float = 2.8284  # ~= 2 * sqrt(2) - two diagonals
    """Used in ``MatchGrouper.filter_unclustered``."""
    CLUSTERING_MIN_SAMPLES: int = 3
    """Used in ``MatchGrouper.filter_unclustered``."""
