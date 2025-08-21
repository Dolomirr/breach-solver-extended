# ################################################################### #
#                                                                     #
# Steps:                                                              #
#   1. Wrap the input into Images container and request the required  #
#       intermediate versions (resized, gray, binary).                #
#   2. Match symbol templates against the global binary image.        #
#   4. If matches are present:                                        #
#       - Group and structure matches into the matrix and daemons.    #
#       - Determine buffer bounds from grouped matches.               #
#       - Extract and binarize the buffer region.                     #
#       - Match buffer templates and count buffer cells.              #
#   5. Build and return SoftTask, summarizing results.                #
#                                                                     #
# ################################################################### #
#                                                                     #
# Pipeline overview:                                                  #
#                                                                     #
#     INPUT: (colored image)                                          #
#         │                                                           #
#         ▼                                                           #
#     ImageProcessor ──► produces resized ─► gray ─► binary images    #
#         │                                                           #
#         ▼                                                           #
#     TemplateLoader ──► loads symbol & buffer templates from disk    #
#         │                                                           #
#         ▼                                                           #
#     TemplateMatcher──► match(binary_image, symbol_templates)        #
#         │                                                           #
#         ▼                                                           #
#     MatchGrouper   ──► cluster & structure matches into:            #
#         │                  • matrix (rows x cols)                   #
#         │                  • daemons (sequences)                    #
#         ▼                                                           #
#     MatchGrouper   ──► find buffer bounds (vertical, horizontal)    #
#         │                                                           #
#         ▼                                                           #
#     ImageProcessor ──► cut buffer region ─► adaptive binarize       #
#         │                                                           #
#         ▼                                                           #
#     TemplateMatcher──► match(buffer_binary, buffer_templates)       #
#         │                                                           #
#         ▼                                                           #
#     OUTPUT: SoftTask(matrix, daemons, buffer_size)                  #
#                                                                     #
# ################################################################### #

import logging

from core import SoftTask, setup_logging

from ..image_loader import ColoredImage
from ..reader_abc import ImageReader
from .match_grouper import MatchGrouper
from .matcher import Match, TemplateMatcher
from .preprocessor import ImageProcessor
from .structs import Images, TemplateProcessingConfig
from .template_loader import TemplateLoader

setup_logging()
log = logging.getLogger(__name__)


class ScannerTemplates(ImageReader[TemplateProcessingConfig]):
    """
    High-level API for template-based scanning.

    Provides single entrypoint for the template-matching based approach of scanning breach protocol mini game.
    It accepts a colored image, runs the templating pipeline, group and filter matches,
    locate the buffer region, detect buffer cells and produces mutable, prepared for further alterations
    result object describing the found matrix, daemon sequences and buffer size.

    It orchestrates components and ensures correct order of operations.
    
    Attributes:
        img_manager : ImageProcessor
            Manipulates different versions of image to make them suitable for next operations.
        templates : TemplateLoader
            Loads and stores template archives from disk; loading is performed on this class init by calling ``templates.read``
             and may raise ``FileNotFoundError`` if templates are missing or corrupted.
        matcher : TemplateMatcher
            Performs template matching.
        grouper : MatchGrouper | None
            Populated during ``read``; groups and structures raw template matches into
            the matrix and daemon sequences.

    Methods:
        read: Main method.
            - Input: single colored image (BGR/uint8).
            - Output: ``SoftTask`` describing:
                - `matrix` - nested list of labels (rows x cols) representing the game matrix,
                - `daemons` - nested list of labels representing daemon sequences,
                - `buffer_size` - integer count of detected buffer cells.
    
    .. seealso::
        :module:`reader.image_loader` and
        :class:`reader.template_matching.structs.TemplateProcessingConfig`
    
    .. important::
        Class is not thread-safe, require creating septate instance for concurrent scanning.
    
    Examples:
    >>> from reader import ScannerTemplates, image_loader
    >>> path = pathlib.Path("/path/to/image_to_scan.png")
    >>> image = image_loader.from_path(path)
    >>> scanner = ScannerTemplates()
    >>> soft_task = scanner.read(image)

    :param config: Immutable configuration used for all pipeline steps.
    :type config: TemplateProcessingConfig
    :raises FileNotFoundError: When required template archives are missing or corrupted.
    
    """
    
    def __init__(self, config: TemplateProcessingConfig | None = None) -> None:
        # This was intentional implemented as class with single method to avoid
        # constant re-reading from disk or storing it globally,
        # since templates are not supposed to be changed between runs
        self.config = (
            config
            if config is not None
            else TemplateProcessingConfig()
            )  # fmt: skip

        self.img_manager = ImageProcessor(self.config)
        self.templates = TemplateLoader(self.config).load()
        self.matcher = TemplateMatcher(self.config)

        log.debug("Initializing ScannerTemplates", extra={"config": config})

    def read(self, image: ColoredImage) -> SoftTask:
        """
        Main method of processing image.
        
        Runs full pipeline of identifying matrix, daemons sequences and counting available buffer cells.
        
        .. warning::
            Calling :meth:`read` is not thread safe on same instance of :class:`ScannerTemplates`
        
        :param image: accept colored image as numpy array loaded from with ``reader.image_loader.from_path``
        :type image: ndarray[tuple[int, int, int], dtype[uint8]]
        :return: mutable type of task
        :rtype: core.soft_task.SofTask
        :raises RuntimeError: Should not happen during normal flow, raised then some elements were altered,
            causing some elements be accessed before assigning.
        """
        self.images = Images(image)
        self.img_manager.set_base(self.images)
        (
            self.img_manager
            .set_resized()
            .set_grayed()
            .set_binary()
        )  # fmt: skip

        log.debug("Matching symbols")
        symbols_matches = self.matcher.match(
            self.images.binary,
            self.templates.symbols,
        )

        if not symbols_matches:
            return SoftTask([[]], [[]], 0)

        self.grouper = MatchGrouper(symbols_matches, self.config)

        (
            self.grouper
            .filter_unclustered()
            .set_splitted()
            .structure_matrix()
            .structure_daemons()
        )  # fmt: skip

        # This cut needed for better dynamic threshold applying on buffer cells
        # that ensures correct matching on wider range of screens brightnesses,
        # while preserving low noise level on regions with symbols.
        buffer_vert_bound, buffer_hor_bound = self.grouper.find_buffer_bounds()
        (
            self.img_manager
            .set_buffer(buffer_vert_bound, buffer_hor_bound)
            .set_buffer_binary()
        )  # fmt: skip

        log.debug("Matched buffer")
        buffer_matches = self.matcher.match(
            self.images.buffer_binary,
            self.templates.buffer,
        )

        return SoftTask(
            matrix=MatchGrouper.extract_labels(self.grouper.matches_matrix),
            daemons=MatchGrouper.extract_labels(self.grouper.matches_daemons),
            buffer_size=len(buffer_matches),
        )
