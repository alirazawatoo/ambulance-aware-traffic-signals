"""Write a tiny synthetic traffic clip for exercising /predict.

The frames are drawn shapes only (a road, lane markings and two moving
blocks), so the clip contains no people, number plates or real footage.

    python scripts/make_synthetic_clip.py samples/synthetic-traffic.avi
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 180


def draw_frame(index: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (70, 70, 70), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (WIDTH, 40), (60, 140, 60), -1)
    for x in range(-40, WIDTH, 40):
        offset = (x + index * 6) % (WIDTH + 40) - 20
        cv2.line(frame, (offset, 110), (offset + 20, 110), (230, 230, 230), 2)
    van_x = 10 + index * 12
    cv2.rectangle(frame, (van_x, 60), (van_x + 70, 100), (245, 245, 245), -1)
    cv2.rectangle(frame, (van_x + 30, 70), (van_x + 40, 90), (40, 40, 220), -1)
    cv2.rectangle(frame, (van_x + 25, 75), (van_x + 45, 85), (40, 40, 220), -1)
    car_x = WIDTH - 60 - index * 8
    cv2.rectangle(frame, (car_x, 125), (car_x + 50, 155), (180, 110, 30), -1)
    return frame


def write_clip(path: Path, frames: int = 12, fps: float = 6.0) -> Path:
    """Write an MJPG AVI, which every OpenCV build can encode and decode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {path}")
    try:
        for index in range(frames):
            writer.write(draw_frame(index))
    finally:
        writer.release()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="destination .avi path")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    print(write_clip(args.output, frames=args.frames))


if __name__ == "__main__":
    main()
