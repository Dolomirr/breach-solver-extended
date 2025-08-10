from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemplateProcessingConfig:
    TARGET_SIZE: tuple[int, int] = (2048, 1080)
    MINVAL_THRESHOLD: int = 0
    MAXVAL_THRESHOLD: int = 1
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
    BUFFER_TEMPLATES: str = "BUFFER_CELL"
    ADDITIONAL_TEMPLATES: frozenset[str] = frozenset()  # not used currently, maybe something later
    MATCHING_THRESHOLD: float = 0.65
    OVERLAP_THRESHOLD: float = 0.01
    CLUSTERING_EPS: float | None = None
    CLUSTERING_EPS_FACTOR: float = 2.8284  # ~= 2 * sqrt(2) - two diagonals
    CLUSTERING_MIN_SAMPLES: int = 3
