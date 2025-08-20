import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np

from core import setup_logging

from .structs import TemplateProcessingConfig

setup_logging()
log = logging.getLogger(__name__)


type Template[S] = np.ndarray[tuple[S, S], np.dtype[np.uint8]] # type: ignore generic type object/int
type SymbolTemplate = Template[Literal[32]]
type BufferTemplate = Template[Literal[40]]
type AdditionalTemplate = Template[int]
type TemplateDict[T] = Mapping[str, tuple[T, ...]]


class TemplateLoader:
    """
    Responsible for loading and storing template data from `.npz` files into corresponding dictionaries.
    
    Attributes:
        folder (pathlib.Path): directory in which archives are stored
        symbols (TemplateDict[SymbolTemplate]): after loading contain a dict with tuple of loaded templates as ``numpy.array``s
        buffer (TemplateDict[BufferTemplate]):
        additional (TemplateDict[AdditionalTemplate]):
    
    Methods:
        load: Loads template data from `.npz` files into dictionaries.
    
    Types:
        Template (np.ndarray[tuple[S, S], np.dtype[np.uint8]]): grayscale, binarized image as array, where `S` is size of each template.
        SymbolTemplate (Template[Literal[32]]):
        BufferTemplate (Template[Literal[40]]):
        AdditionalTemplate (Template[int]):
        TemplateDict[T] (Mapping[str, tuple[T, ...]]): dict with label and tuple of multiple templates, where `T` is type of template.
    
    :param config:
    :type config: TemplateProcessingConfig
    :param subdir: Subdirectory where the templates are located.
        Default: `"templates"`.
    :type subdir: str

    """
    
    folder: Path
    symbols: TemplateDict[SymbolTemplate]
    buffer: TemplateDict[BufferTemplate]
    additional: TemplateDict[AdditionalTemplate]
    """Currently unused, left for possible additional information that may be scanned with template matching."""

    def __init__(self, config: TemplateProcessingConfig, subdir: str = "templates") -> None:
        self.config = config

        folder = Path(__file__).parent / subdir

        if not folder.exists():
            msg = f"Templates directory is not found: {folder}"
            log.exception(msg)
            raise FileNotFoundError(msg)
        self.folder = folder

    def load(self) -> Self:
        """
        Loads templates data from `.npz` to the corresponding dictionaries.

        Templates archives loaded from defined on init subdirectory.
        Each archive contains few variants of same symbol used in Breach Protocol.
        
        Type od templates (label) stored in archive is decided by its name.
            - base game symbols and dlc symbols: 1C, 55, BD, E9, 7A, FF, X9, XX, XH, IX, XR
            - buffer cell: BUFFER_CELL
            - additional templates: currently none, all newly added needs to be defined in ``TemplateProcessingConfig.ADDITIONAL_TEMPLATES``
        
        Uses:
            :attr:`config.BUFFER_TEMPLATES`
            :attr:`config.EXISTING_TEMPLATES`
            :attr:`config.ADDITIONAL_TEMPLATES` if set
        
        :raises RuntimeError: if some error accuses during loading.
        :raises: FileNotFoundError: if some template ise defined by missing in defined subdirectory.
        """
        templates: TemplateDict[SymbolTemplate] = {}
        buffer_templates: TemplateDict[BufferTemplate] = {}

        # currently unused, just in case of need to add new non-standard symbols.
        additional_templates: TemplateDict[AdditionalTemplate] = {}

        buffer_template_found = False

        # TODO!: this need to be refactored + additional templates should actually be used lmao
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
        
        # checking for additional

        self.symbols = templates
        self.buffer = buffer_templates
        self.additional = additional_templates
        log.info("Successfully loaded templates")
        return self
