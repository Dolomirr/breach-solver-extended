import logging
from collections import defaultdict, deque
from typing import Literal, Self, cast

import numpy as np

from core import setup_logging

from .matcher import Match, NullMatch
from .structs import TemplateProcessingConfig

setup_logging()
log = logging.getLogger(__name__)

type Array1DIndices = np.ndarray[tuple[int], np.dtype[np.integer]]


class MatchGrouper:
    """
    Group and structure Match objects based on their spatial relationships.

    Attributes:
        config (TemplateProcessingConfig): Configuration settings for clustering.
        matches (list[Match]): List of all Match objects.
        matches_matrix (list[list[Match | NullMatch]]): Structured 2D list of Match objects for the matrix.
        matches_daemons (list[list[Match]]): Structured 2D list of Match objects for the daemons.

    Methods:
        filter_unclustered: Filter out Matches that are to far away from main clusters.
        set_splitted: Splits list of Matches into two groups between matrix and daemons based on their x-axis center coordinates.
        structure_matrix: Structure matches in the matrix according to real 'matrix' structure.
        structure_daemons: Structure matches in the daemons according to real 'sequences' structure.
        find_buffer_bounds: Find buffer bounds for the structured Match objects.
        extract_labels: Extract labels from a 2D list of Match objects.

    """

    config: TemplateProcessingConfig

    matches: list[Match]
    _matches_matrix_flat: list[Match]
    _matches_daemons_flat: list[Match]
    matches_matrix: list[list[Match | NullMatch]]
    matches_daemons: list[list[Match]]

    def __init__(self, matches: list[Match], config: TemplateProcessingConfig) -> None:
        self.matches = matches.copy()
        self.config = config

    def filter_unclustered(self) -> Self:
        """
        Filters out points not belonging to any cluster (matrix/sequences) with simple DBSCAN.

        Intended to filter random matches that are too far from any other point. Result,
        however, require further filtering by structure (matrix/daemons rows and columns)

        Uses:
            :attr:`config.CLUSTERING_EPS` if set
            :attr:`config.CLUSTERING_EPS_FACTOR` overwise to determine epsilon for clustering.
            :attr:`config.CLUSTERING_MIN_SAMPLES` as minimal population of single cluster.
        """
        len_before = len(self.matches)

        epsilon = (
            self.config.CLUSTERING_EPS
            if self.config.CLUSTERING_EPS is not None
            else (np.mean([m.bbox[2] - m.bbox[0] for m in self.matches]) * self.config.CLUSTERING_EPS_FACTOR)
                ** 2  # square to avoid root lately
        )  # fmt: skip

        points: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = np.array([m.center for m in self.matches])
        n_points = points.shape[0]

        # distance matrix
        diff: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = points[:, np.newaxis] - points[np.newaxis, :]
        sq_dist = np.sum(diff**2, axis=-1)

        neighbors = []
        for i in range(n_points):
            mask = sq_dist[i] <= epsilon
            neighbors.append(np.where(mask)[0])

        labels = np.full(n_points, -1, dtype=np.int32)  # -1: unvisited, 0: noise, >0: cluster id
        cluster_id = 0
        queue = deque()

        for i in range(n_points):
            if labels[i] != -1:
                continue

            if len(neighbors[i]) < self.config.CLUSTERING_MIN_SAMPLES:
                labels[i] = 0
                continue

            cluster_id += 1
            labels[i] = cluster_id
            queue.extend(neighbors[i])

            while queue:
                j = queue.popleft()

                if labels[j] == 0:
                    labels[j] = cluster_id

                if labels[j] != -1:
                    continue

                labels[j] = cluster_id

                if len(neighbors[j]) >= self.config.CLUSTERING_MIN_SAMPLES:
                    for k in neighbors[j]:
                        if labels[k] <= 0:
                            queue.append(k)

        self.matches = [match for i, match in enumerate(self.matches) if labels[i] > 0]
        log.debug("Filter on clustered:", extra={"before": len_before, "after": len(self.matches)})
        return self

    def _get_centers(self, matches: list[Match]) -> tuple[Array1DIndices, Array1DIndices]:
        """
        Extract sorted, unique x and y center coordinates from a list of Match objects.
        Uniqueness is treated as 'centers are separated by at least half the bbox size on one of axis'.
        Ignores NullMatches (coordinates that lower then 0).

        :returns: tuple[np.ndarray, np.ndarray]
            xs: np.ndarray of unique x-centers (sorted).
            ys: np.ndarray of unique y-centers (sorted).
        """
        if len(matches) == 0:
            msg = "Matches list is empty, what should not happened at that point of execution."
            raise RuntimeError(msg)

        centers = [(match.center[0], match.center[1]) for match in matches]

        bbox_width = matches[0].bbox[2] - matches[0].bbox[0]
        bbox_height = matches[0].bbox[3] - matches[0].bbox[1]
        min_distance = max(bbox_width, bbox_height) // 2

        centers.sort(key=lambda c: (c[0], c[1]))

        unique_centers = []

        for current_center in centers:
            is_unique = True
            for accepted_center in unique_centers:
                if (abs(current_center[0] - accepted_center[0]) < min_distance
                    or abs(current_center[1] - accepted_center[1]) < min_distance):  # fmt: skip
                    is_unique = False
                    break

            if is_unique:
                unique_centers.append(current_center)

        unique_centers.sort(key=lambda c: (c[0], c[1]))

        centers_x = np.array([c[0] for c in unique_centers])
        centers_y = np.array([c[1] for c in unique_centers])

        return cast("Array1DIndices", centers_x), cast("Array1DIndices", centers_y)

    def _find_gaps(
        self,
        centers: np.ndarray[tuple[int], np.dtype[np.integer]],
    ) -> np.ndarray[tuple[int], np.dtype[np.signedinteger]]:
        """
        Finds midpoints of gaps in a sorted array of 1d points.

        :param centers: 1d sorted array of either x coordinates or y coordinates
        :return: array of midpoints of found gaps sorted decreasingly, if to less points to calculate gaps - return empty array.
        """
        diffs = np.diff(centers)

        if diffs.size == 0:
            msg = "Centers array is empty, what should not happened at that point of execution."
            raise RuntimeError(msg)

        gaps_indices = np.argsort(diffs)[::-1]
        gaps_midpoints = (centers[gaps_indices] + centers[gaps_indices + np.intp(1)]) // 2

        log.debug("Gaps found", extra={"gaps_midpoints": gaps_midpoints})
        return cast(
            "np.ndarray[tuple[int], np.dtype[np.signedinteger]]",
            gaps_midpoints,
        )

    def set_splitted(self) -> Self:
        """
        Split the list of Match objects into two groups:
            - matches where center on x axis is strictly less than split_x (matrix)
            - matches where center x is greater than or equal to split_x (daemons)
        """
        centers_x, _ = self._get_centers(self.matches)
        split_x = self._find_gaps(centers_x)[0]

        left = [m for m in self.matches if m.center.cx < split_x]
        right = [m for m in self.matches if m.center.cx >= split_x]

        self._matches_matrix_flat, self._matches_daemons_flat = left, right
        log.debug("Spited matches", extra={"matrix": len(left), "daemons": len(right)})
        return self

    def structure_matrix(self) -> Self:  # noqa: PLR0915
        """
        Filter/structure matches in some godforsaken way according to their real matrix structure.
        Filters "out of square" matches, re-organize flat list to square nested list

        :param matches: flat list of Matches belonging to matrix.
        :return: structured and filtered matches with 2d nested list.
        """
        if not self._matches_matrix_flat:
            self.matches_matrix = [[]]
            log.warning("No matrix matches found")
            return self

        tolerance = (
            self._matches_matrix_flat[0].bbox[2] - self._matches_matrix_flat[0].bbox[0]
        ) // 2  # valid (?) assumption that all matches have same bbox width and all are squares

        xs = [m.center[0] for m in self._matches_matrix_flat]
        xs.sort()
        col_clusters = []
        current_cluster = [xs[0]]
        for i in range(1, len(xs)):
            if xs[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(xs[i])
            else:
                col_clusters.append(current_cluster)
                current_cluster = [xs[i]]
        col_clusters.append(current_cluster)

        ys = [m.center[1] for m in self._matches_matrix_flat]
        ys.sort()
        row_clusters = []
        current_cluster = [ys[0]]
        for i in range(1, len(ys)):
            if ys[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(ys[i])
            else:
                row_clusters.append(current_cluster)
                current_cluster = [ys[i]]
        row_clusters.append(current_cluster)

        row_centers = [np.median(cluster).astype(np.int64) for cluster in row_clusters]
        row_centers.sort()
        col_centers = [np.median(cluster).astype(np.int64) for cluster in col_clusters]
        col_centers.sort()

        grid: list[list[Match | NullMatch]] = [[NullMatch.instance()] * len(col_centers) for _ in range(len(row_centers))]
        for sym_match in self._matches_matrix_flat:
            min_col_dist = np.inf
            best_col_idx = -1
            for i, centers_x in enumerate(col_centers):
                dist_x = abs(sym_match.center[0] - centers_x)
                if dist_x < min_col_dist:
                    min_col_dist = dist_x
                    best_col_idx = i

            min_row_dist = np.inf
            best_row_idx = -1
            for i, centers_y in enumerate(row_centers):
                dist_y = abs(sym_match.center[1] - centers_y)
                if dist_y < min_row_dist:
                    min_row_dist = dist_y
                    best_row_idx = i

            if min_row_dist <= tolerance and min_col_dist <= tolerance:
                existing_match = grid[best_row_idx][best_col_idx]
                if existing_match is NullMatch.instance():
                    grid[best_row_idx][best_col_idx] = sym_match
                else:
                    current_center = (col_centers[best_col_idx], row_centers[best_row_idx])
                    dist_real = (existing_match.center[0] - current_center[0]) ** 2 + (
                        existing_match.center[1] - current_center[1]
                    ) ** 2
                    dist_current = (sym_match.center[0] - current_center[0]) ** 2 + (
                        sym_match.center[1] - current_center[1]
                    ) ** 2
                    if dist_current > dist_real:
                        grid[best_row_idx][best_col_idx] = sym_match

        if any(NullMatch.instance() in row for row in grid):
            msg = "Some of grid cell was not replaced."
            log.debug(msg)
            log.debug(grid)

        if any(None in row for row in grid):
            msg = "Some of grid cell are 'None'."
            log.exception(msg)
            raise RuntimeError(msg)

        self.matches_matrix = cast("list[list[Match | NullMatch]]", grid)
        log.debug(
            "Structured/filtering on matrix",
            extra={
                "before": len(self._matches_matrix_flat),
                "after": sum(len(row) for row in grid),
            },
        )
        return self

    def structure_daemons(self) -> Self:
        """
        Filter/structure matches in some godforsaken way according to their 'sequence' structure.
        Filters "out of square" matches, re-organize flat list to nested list

        :param matches: List of Matches belonging to daemons
        """
        if not self._matches_daemons_flat:
            self.matches_daemons = [[]]
            log.warning("No daemons matches found")
            return self

        tolerance = (self._matches_daemons_flat[0].bbox[2] - self._matches_daemons_flat[0].bbox[0]) // 2

        sequences = []
        used = set()
        for tmp_match in self._matches_daemons_flat:
            if tmp_match in used:
                continue
            group = []
            q = deque([tmp_match])
            used.add(tmp_match)
            while q:
                current = q.popleft()
                group.append(current)
                for candidate in self._matches_daemons_flat:
                    if candidate in used:
                        continue
                    if abs(current.center[1] - candidate.center[1]) <= tolerance:
                        used.add(candidate)
                        q.append(candidate)
            sequences.append(group)

        sequences = [
            sorted(sequences[i], key=lambda m: m.center[0])
            for i in range(len(sequences))
            ]  # fmt: skip

        starts_x = [group[0].center[0] for group in sequences if group]

        if not starts_x:
            self.matches_daemons = [[]]
            log.warning("No daemons matches found")
            return self

        # can we assume no invalid matches will be on the left side?
        x_main = np.median(starts_x).astype(np.int64)

        valid_rows = [
            group for group in sequences
            if group and abs(group[0].center[0] - x_main) <= tolerance
        ]  # fmt: skip

        cols = defaultdict(list)
        for row in valid_rows:
            for i, m in enumerate(row):
                cols[i].append(m.center[0])

        cols_medians = {
            idx: (np.median(xs).astype(np.int64) if len(xs) > 2 else None)
            for idx, xs in cols.items()
        }  # fmt: skip

        valid_sequences = []
        for row in valid_rows:
            new_row = []
            for j, m in enumerate(row):
                median = cols_medians[j]
                if median is None or abs(m.center[0] - cols_medians[j] <= tolerance):
                    new_row.append(m)
            if new_row:
                valid_sequences.append(new_row)

        valid_sequences.sort(key=lambda row: min(m.center[1] for m in row))

        self.matches_daemons = valid_sequences
        log.debug(
            "Structured/filtering on matrix",
            extra={
                "before": len(self._matches_daemons_flat),
                "after": sum(len(row) for row in valid_sequences),
            },
        )
        return self

    def find_buffer_bounds(self) -> tuple[int, int]:
        """
        Based on filtered and structured matches finds bound of region where buffer cells are located.
        
        :returns: coordinates of vertical bound, and coordinates of horizontal bound.
        """
        matches_after_filtering = [
            item for sublist
                in (self.matches_matrix + self.matches_daemons)
                    for item in sublist
            ]  # fmt: skip

        centers_filtered_x, _ = self._get_centers(matches_after_filtering)
        gap_filtered = self._find_gaps(centers_filtered_x)[0]
        upper_filtered = max(m.center.cy for m in matches_after_filtering)

        log.debug("Located buffer bounds.", extra={"vert_bound": gap_filtered, "hor_bound": upper_filtered})
        return gap_filtered, upper_filtered

    @staticmethod
    def extract_labels(matches: list[list[Match]]):
        return [[cell.label for cell in row] for row in matches]
