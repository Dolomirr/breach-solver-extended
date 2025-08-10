import logging
from pathlib import Path

from core import SoftTask, setup_logging

from ..image_loader import ColoredImage
from ..reader_abc import ImageReader
from .match_grouper import MatchGrouper
from .matcher import TemplateMatcher
from .preprocessor import ImageProcessor
from .structs import Images, TemplateProcessingConfig
from .template_loader import TemplateLoader

setup_logging()
log = logging.getLogger(__name__)


class ScannerTemplates(ImageReader[TemplateProcessingConfig]):
    def __init__(self, config: TemplateProcessingConfig | None = None) -> None:
        self.config = (
            config
            if config is not None
            else TemplateProcessingConfig()
            )  # fmt: skip

        self.img_manager = ImageProcessor(self.config)
        self.templates = TemplateLoader(self.config)
        self.matcher = TemplateMatcher(self.config)

        log.debug("Initializing ScannerTemplates", extra={"config": config})

    def read(self, image: ColoredImage) -> SoftTask:
        self.images = Images(image)
        self.img_manager.set_base(self.images)
        (
            self.img_manager
            .set_resized()
            .set_grayed()
            .set_binary()
        )  # fmt: skip

        # TODO?: quick return if None here?
        self.templates.load()
        
        log.debug('Matching symbols')
        symbols_matches = self.matcher.match(
            self.images.binary,
            self.templates.symbols,
        )

        self.grouper = MatchGrouper(symbols_matches, self.config)

        (
            self.grouper
            .filter_unclustered()
            .set_splitted()
            .structure_matrix()
            .structure_daemons()
        )  # fmt: skip

        buffer_vert_bound, buffer_hor_bound = self.grouper.find_buffer_bounds()

        (
            self.img_manager
            .set_buffer(buffer_vert_bound, buffer_hor_bound)
            .set_buffer_binary()
        )  # fmt: skip

        log.debug('Matched buffer')
        buffer_matches = self.matcher.match(
            self.images.buffer_binary,
            self.templates.buffer,
        )

        return SoftTask(
            matrix=[[cell.label for cell in row] for row in self.grouper.matches_matrix],
            daemons=[[cell.label for cell in row] for row in self.grouper.matches_daemons],
            buffer_size=len(buffer_matches),
        )
