from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from .detector import ChessboardDetector, DetectionError
from .render import draw_detection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a chessboard and draw its 8x8 grid on the input image."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--output",
        "-o",
        help="Path for the rendered overlay image",
        default="board_overlay.jpg",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.15,
        help="Minimum acceptable candidate score for fallback detection",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}", file=sys.stderr)
        return 2

    detector = ChessboardDetector()
    try:
        detection = detector.detect(image, min_score=args.min_score)
    except DetectionError as exc:
        print(f"Detection failed: {exc}", file=sys.stderr)
        return 1

    overlay = draw_detection(image, detection)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    print(f"Wrote overlay to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
