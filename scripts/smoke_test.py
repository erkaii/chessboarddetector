from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chessboarddetector.detector import ChessboardDetector
from chessboarddetector.render import draw_detection


def make_synthetic_board(size: int = 900) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 60
            board[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = color

    for col in range(8):
        center = (col * cell + cell // 2, 2 * cell + cell // 2)
        cv2.circle(board, center, cell // 3, (20, 20, 20), -1)
    for col in range(8):
        center = (col * cell + cell // 2, 5 * cell + cell // 2)
        cv2.circle(board, center, cell // 3, (235, 235, 235), -1)

    src = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )
    dst = np.array(
        [[140, 120], [820, 30], [860, 870], [40, 760]], dtype=np.float32
    )
    canvas = np.full((1000, 1000, 3), 30, dtype=np.uint8)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, matrix, (1000, 1000))
    mask = cv2.warpPerspective(
        np.full((size, size), 255, dtype=np.uint8), matrix, (1000, 1000)
    )
    canvas[mask > 0] = warped[mask > 0]
    return canvas


def main() -> int:
    image = make_synthetic_board()
    detector = ChessboardDetector()
    detection = detector.detect(image, min_score=0.05)
    overlay = draw_detection(image, detection)

    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "synthetic_input.jpg"), image)
    cv2.imwrite(str(output_dir / "synthetic_overlay.jpg"), overlay)

    print(detection.method, f"{detection.score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
