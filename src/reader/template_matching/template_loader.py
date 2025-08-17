import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import numpy as np

from core import setup_logging

from .structs import TemplateProcessingConfig

setup_logging()
log = logging.getLogger(__name__)

type SymbolTemplate = np.ndarray[tuple[Literal[32], Literal[32]], np.dtype[np.uint8]]
type BufferTemplate = np.ndarray[tuple[Literal[40], Literal[40]], np.dtype[np.uint8]]
type AdditionalTemplate = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type TemplateDict[T] = Mapping[str, tuple[T, ...]]


class TemplateLoader:
    folder: Path
    symbols: TemplateDict[SymbolTemplate]
    buffer: TemplateDict[BufferTemplate]
    additional: TemplateDict[AdditionalTemplate]
    """Currently unused, maybe add something later"""

    def __init__(self, config: TemplateProcessingConfig, subdir: str = "templates") -> None:
        self.config = config

        folder = Path(__file__).parent / subdir

        if not folder.exists():
            msg = f"Templates directory is not found: {folder}"
            log.exception(msg)
            raise FileNotFoundError(msg)
        self.folder = folder

    def load(self) -> None:
        templates: TemplateDict[SymbolTemplate] = {}
        buffer_templates: TemplateDict[BufferTemplate] = {}

        # currently unused, just in case of need to add new non-standard symbols.
        additional_templates: TemplateDict[AdditionalTemplate] = {}

        buffer_template_found = False

        for npz_path in self.folder.glob("*.npz"):
            label = npz_path.stem

            if label == self.config.BUFFER_TEMPLATES:
                try:
                    data = np.load(npz_path, allow_pickle=False)
                    tmpls = tuple(data[f"{label}_{i}"] for i in range(len(data.files)))
                    buffer_templates[label] = cast("tuple[BufferTemplate, ...]", tmpls)
                    buffer_template_found = True
                except Exception as e:
                    msg = "Error loading template, some templates may be corrupted."
                    log.exception(msg, extra={"on": "BUFFER"})
                    raise RuntimeError(msg) from e

            elif label in self.config.EXISTING_TEMPLATES:
                try:
                    data = np.load(npz_path, allow_pickle=False)
                    tmpls = tuple(data[f"{label}_{i}"] for i in range(len(data.files)))
                    templates[label] = cast("tuple[SymbolTemplate, ...]", tmpls)
                except Exception as e:
                    msg = "Error loading template, some templates may be corrupted."
                    log.exception(msg, extra={"on": "TEMPLATES"})
                    raise RuntimeError(msg) from e

            else:
                try:
                    data = np.load(npz_path, allow_pickle=False)
                    tmpls = tuple(data[f"{label}_{i}"] for i in range(len(data.files)))
                    additional_templates[label] = cast("tuple[SymbolTemplate, ...]", tmpls)
                except Exception as e:
                    msg = "Error loading template, some templates may be corrupted."
                    log.exception(msg, extra={"on": "EXTRA"})
                    raise RuntimeError(msg) from e

        if not buffer_template_found:
            msg = "Buffer templates are corrupted or missing."
            log.exception(msg)
            raise FileNotFoundError(msg)

        if len(templates) != len(self.config.EXISTING_TEMPLATES):
            print("TEMPLATES")
            missing = set(self.config.EXISTING_TEMPLATES) - set(templates.keys())
            msg = "Some templates are corrupted or missing."
            log.exception(msg, extra={"missing": missing})
            raise FileNotFoundError(msg)

        self.symbols = templates
        self.buffer = buffer_templates
        self.additional = additional_templates
        log.info("Successfully loaded templates")
