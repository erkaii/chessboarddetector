# Chessboard Detector

This repository contains a pure computer-vision chessboard detector written in Python with OpenCV. It takes an image of a chessboard, estimates the board pose even under perspective distortion and partial occlusion from pieces, and writes out an image with the full 8x8 grid drawn on top.

## Approach

The detector uses a multi-stage pipeline:

1. Try OpenCV's `findChessboardCornersSB` on the `7x7` inner-corner pattern across several grayscale variants, including inverted and contrast-normalized views.
2. If the full-image helper fails, generate board quadrilateral candidates from contours and oriented rectangles.
3. Warp each candidate to a square view and retry `findChessboardCornersSB` on the rectified board crop before falling back to geometric scoring.
4. Score remaining candidates using:
   - contrast between neighboring cells in the expected checker pattern,
   - edge energy along expected internal grid lines,
   - consistency of color variation across rows and columns.
5. Pick the best result and project the `9x9` lattice back onto the original image.

This keeps the implementation in classical CV only: no training, no neural nets.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
chessboarddetector path/to/input.jpg --output output/board_overlay.jpg
```

Optional arguments:

```bash
chessboarddetector input.jpg \
  --output output.jpg \
  --min-score 0.15
```

## Output

- `--output`: original image with the detected board border and the 8x8 squares drawn.

If detection fails, the CLI exits with a non-zero status and prints the best score it found.
