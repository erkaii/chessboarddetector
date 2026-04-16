from __future__ import annotations

import cv2
import numpy as np

from .detector import BoardDetection


def draw_detection(image: np.ndarray, detection: BoardDetection) -> np.ndarray:
    canvas = image.copy()
    corners = detection.corners.astype(int)

    cv2.polylines(canvas, [corners.reshape(-1, 1, 2)], True, (0, 255, 255), 3)

    grid = detection.grid_points
    for row in range(9):
        pts = np.round(grid[row]).astype(int).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (0, 180, 255), 2)

    for col in range(9):
        pts = np.round(grid[:, col]).astype(int).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (0, 180, 255), 2)

    label = f"{detection.method} score={detection.score:.3f}"
    cv2.putText(
        canvas,
        label,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )
    return canvas
