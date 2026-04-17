from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class DetectionError(RuntimeError):
    pass


@dataclass
class BoardDetection:
    corners: np.ndarray
    grid_points: np.ndarray
    score: float
    method: str


class ChessboardDetector:
    def __init__(self, warp_size: int = 800) -> None:
        self.warp_size = warp_size

    def detect(self, image: np.ndarray, min_score: float = 0.15) -> BoardDetection:
        if image is None or image.size == 0:
            raise DetectionError("Input image is empty.")

        gray = self._preprocess(image)

        helper_result = self._detect_with_opencv_helper(gray, image.shape)
        if helper_result is not None:
            return helper_result

        candidates = self._generate_candidates(gray)
        if not candidates:
            raise DetectionError("No board candidates were found.")

        scored_candidates: list[BoardDetection] = []
        for candidate in candidates:
            scored = self._score_candidate(image, gray, candidate)
            if scored is None:
                continue
            scored_candidates.append(scored)

        if not scored_candidates:
            raise DetectionError("No valid board candidate survived scoring.")

        scored_candidates.sort(key=lambda detection: detection.score, reverse=True)
        best_detection = scored_candidates[0]

        helper_retry_count = min(5, len(scored_candidates))
        for detection in scored_candidates[:helper_retry_count]:
            refined = self._detect_helper_on_candidate(image, detection.corners)
            if refined is not None:
                return refined

        if best_detection.score < min_score:
            raise DetectionError(
                f"Best candidate score {best_detection.score:.3f} is below threshold {min_score:.3f}."
            )
        return best_detection

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _helper_variants(self, gray: np.ndarray) -> list[np.ndarray]:
        clahe_strong = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(gray)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )

        variants = [
            gray,
            255 - gray,
            clahe_strong,
            255 - clahe_strong,
            normalized,
            255 - normalized,
            sharpened,
            255 - sharpened,
            adaptive,
            255 - adaptive,
        ]

        deduped: list[np.ndarray] = []
        seen: set[bytes] = set()
        for variant in variants:
            clipped = np.clip(variant, 0, 255).astype(np.uint8)
            key = clipped.tobytes()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(clipped)
        return deduped

    def _detect_with_opencv_helper(
        self, gray: np.ndarray, image_shape: tuple[int, ...]
    ) -> BoardDetection | None:
        return self._run_helper(gray, image_shape, method_prefix="findChessboardCornersSB")

    def _detect_helper_on_candidate(
        self, image: np.ndarray, candidate: np.ndarray
    ) -> BoardDetection | None:
        dst = np.array(
            [
                [0, 0],
                [self.warp_size - 1, 0],
                [self.warp_size - 1, self.warp_size - 1],
                [0, self.warp_size - 1],
            ],
            dtype=np.float32,
        )
        candidate = candidate.astype(np.float32)
        image_to_warp = cv2.getPerspectiveTransform(candidate, dst)
        board_view = cv2.warpPerspective(image, image_to_warp, (self.warp_size, self.warp_size))
        gray = self._preprocess(board_view)

        helper_detection = self._run_helper(
            gray,
            board_view.shape,
            method_prefix="findChessboardCornersSB_rectified",
        )
        if helper_detection is None:
            return None

        board_to_image = cv2.getPerspectiveTransform(dst, candidate)
        warped_grid = helper_detection.grid_points.reshape(-1, 1, 2).astype(np.float32)
        image_grid = cv2.perspectiveTransform(warped_grid, board_to_image).reshape(9, 9, 2)
        warped_corners = helper_detection.corners.reshape(-1, 1, 2).astype(np.float32)
        image_corners = cv2.perspectiveTransform(warped_corners, board_to_image).reshape(4, 2)

        return BoardDetection(
            corners=self._order_corners(image_corners),
            grid_points=image_grid,
            score=1.0,
            method=helper_detection.method,
        )

    def _run_helper(
        self, gray: np.ndarray, image_shape: tuple[int, ...], method_prefix: str
    ) -> BoardDetection | None:
        flags = (
            cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        template_inner = np.array(
            [[x, y] for y in range(1, 8) for x in range(1, 8)], dtype=np.float32
        )
        outer_template = np.array(
            [[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32
        ).reshape(-1, 1, 2)

        for index, variant in enumerate(self._helper_variants(gray)):
            found, corners = cv2.findChessboardCornersSB(variant, (7, 7), flags=flags)
            if not found or corners is None or len(corners) != 49:
                continue

            inner = corners.reshape(-1, 2).astype(np.float32)
            homography, _ = cv2.findHomography(template_inner, inner, cv2.RANSAC, 3.0)
            if homography is None:
                continue

            projected_outer = cv2.perspectiveTransform(
                outer_template, homography
            ).reshape(-1, 2)
            if not self._corners_inside_image(projected_outer, image_shape):
                continue

            grid_points = self._project_full_grid(homography)
            method = method_prefix
            if index > 0:
                method = f"{method_prefix}_variant{index}"
            return BoardDetection(
                corners=self._order_corners(projected_outer),
                grid_points=grid_points,
                score=1.0,
                method=method,
            )

        return None

    def _generate_candidates(self, gray: np.ndarray) -> list[np.ndarray]:
        edges = cv2.Canny(gray, 50, 150)
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=2,
        )
        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        image_area = gray.shape[0] * gray.shape[1]
        candidates: list[np.ndarray] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.05:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                candidates.append(self._order_corners(quad))

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            candidates.append(self._order_corners(box))

        if not candidates:
            return []

        deduped: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for candidate in candidates:
            key = tuple(np.round(candidate.reshape(-1) / 8.0).astype(int).tolist())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _score_candidate(
        self, image: np.ndarray, gray: np.ndarray, candidate: np.ndarray
    ) -> BoardDetection | None:
        dst = np.array(
            [
                [0, 0],
                [self.warp_size - 1, 0],
                [self.warp_size - 1, self.warp_size - 1],
                [0, self.warp_size - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(candidate.astype(np.float32), dst)
        warp_gray = cv2.warpPerspective(gray, transform, (self.warp_size, self.warp_size))
        warp_bgr = cv2.warpPerspective(
            image, transform, (self.warp_size, self.warp_size)
        )

        cell_means = self._cell_means(warp_gray)
        contrast_score = self._checker_contrast_score(cell_means)
        line_score = self._grid_line_score(warp_gray)
        stability_score = self._row_col_variation_score(cell_means)
        score = 0.55 * contrast_score + 0.30 * line_score + 0.15 * stability_score

        if not np.isfinite(score):
            return None

        board_coords = np.array(
            [[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32
        )
        board_to_image = cv2.getPerspectiveTransform(
            board_coords, candidate.astype(np.float32)
        )
        grid_points = self._project_full_grid(board_to_image)

        return BoardDetection(
            corners=candidate,
            grid_points=grid_points,
            score=float(score),
            method="candidate_scoring",
        )

    def _cell_means(self, warp_gray: np.ndarray) -> np.ndarray:
        cells = np.zeros((8, 8), dtype=np.float32)
        step = warp_gray.shape[0] / 8.0
        for row in range(8):
            for col in range(8):
                y0 = int(round(row * step + step * 0.2))
                y1 = int(round((row + 1) * step - step * 0.2))
                x0 = int(round(col * step + step * 0.2))
                x1 = int(round((col + 1) * step - step * 0.2))
                patch = warp_gray[y0:y1, x0:x1]
                cells[row, col] = float(np.median(patch))
        return cells

    def _checker_contrast_score(self, cell_means: np.ndarray) -> float:
        pattern = np.fromfunction(lambda y, x: (x + y) % 2, (8, 8), dtype=int)
        score = 0.0
        for candidate in (pattern, 1 - pattern):
            dark = cell_means[candidate == 0]
            light = cell_means[candidate == 1]
            score = max(score, float(np.median(light) - np.median(dark)))
        return float(np.clip(score / 80.0, 0.0, 1.0))

    def _grid_line_score(self, warp_gray: np.ndarray) -> float:
        grad_x = cv2.Sobel(warp_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(warp_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        step = warp_gray.shape[0] / 8.0

        samples: list[float] = []
        for index in range(1, 8):
            offset = int(round(index * step))
            band = 4
            vertical_strip = mag[:, max(0, offset - band) : min(mag.shape[1], offset + band)]
            horizontal_strip = mag[max(0, offset - band) : min(mag.shape[0], offset + band), :]
            samples.append(float(np.mean(vertical_strip)))
            samples.append(float(np.mean(horizontal_strip)))

        baseline = float(np.mean(mag))
        if baseline < 1e-6:
            return 0.0
        return float(np.clip((np.median(samples) / baseline - 1.0) / 2.5, 0.0, 1.0))

    def _row_col_variation_score(self, cell_means: np.ndarray) -> float:
        row_std = np.mean(np.std(cell_means, axis=1))
        col_std = np.mean(np.std(cell_means, axis=0))
        return float(np.clip((row_std + col_std) / 90.0, 0.0, 1.0))

    def _project_full_grid(self, homography: np.ndarray) -> np.ndarray:
        template = np.array(
            [[x, y] for y in range(9) for x in range(9)], dtype=np.float32
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(template, homography).reshape(9, 9, 2)
        return projected

    def _corners_inside_image(
        self, corners: np.ndarray, image_shape: tuple[int, ...], margin: float = 0.1
    ) -> bool:
        height, width = image_shape[:2]
        x_ok = np.all(
            (corners[:, 0] >= -width * margin) & (corners[:, 0] <= width * (1.0 + margin))
        )
        y_ok = np.all(
            (corners[:, 1] >= -height * margin)
            & (corners[:, 1] <= height * (1.0 + margin))
        )
        return bool(x_ok and y_ok)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        corners = np.asarray(corners, dtype=np.float32)
        sums = corners.sum(axis=1)
        diffs = corners[:, 0] - corners[:, 1]
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(sums)]
        ordered[2] = corners[np.argmax(sums)]
        ordered[1] = corners[np.argmin(diffs)]
        ordered[3] = corners[np.argmax(diffs)]
        return ordered
