import logging

import cv2
import numpy as np

from core import setup_logging
from image_reader.image_loader import GrayScaleImage

from ..structs import Images, TemplateProcessingConfig
from ..template_loader import AdditionalTemplate, BufferTemplate, SymbolTemplate, TemplateDict
from .match_struct import BBox, Center, Match

setup_logging()
log = logging.getLogger(__name__)


class TemplateMatcher:
    def __init__(self, config: TemplateProcessingConfig) -> None:
        self.config = config

    def match(
        self,
        image: GrayScaleImage,
        templates: TemplateDict[SymbolTemplate | BufferTemplate | AdditionalTemplate],
    ) -> list[Match]:
        raw: list[tuple[int, int, int, int, float, str, int]] = []  # [(x, y, w, h, score, label, template_idx), ...]
        for label, tmpl_list in templates.items():
            for idx, tmpl in enumerate(tmpl_list):
                h, w = tmpl.shape[:2]
                res = cv2.matchTemplate(
                    image,
                    tmpl,
                    cv2.TM_CCOEFF_NORMED,
                )
                ys, xs = np.where(res >= self.config.MATCHING_THRESHOLD)
                for x, y in zip(xs, ys, strict=True):
                    score = float(res[y, x])
                    raw.append((x, y, w, h, score, label, idx))

        if not raw:
            log.warning('No matches found. Returning empty list.')
            return []

        boxes = np.array([entry[:4] for entry in raw])  # (x, y, w, h)
        scores = np.array([entry[4] for entry in raw], dtype=float)

        keep_idx = cv2.dnn.NMSBoxes(
            bboxes=boxes,  # type: ignore (arrays are not recognized as sequence)
            scores=scores,  # type: ignore
            score_threshold=self.config.MATCHING_THRESHOLD,
            nms_threshold=self.config.OVERLAP_THRESHOLD,
        )

        matches: list[Match] = []
        for i in keep_idx:
            x, y, w, h = boxes[i]
            score, label, tmpl_idx = raw[i][4], raw[i][5], raw[i][6]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            matches.append(
                Match(
                    label=label,
                    template_idx=tmpl_idx,
                    score=score,
                    bbox=BBox(x1, y1, x2, y2),
                    center=Center(cx, cy),
                ),
            )

        log.debug("Template matching", extra={"found": len(matches)})
        return matches
