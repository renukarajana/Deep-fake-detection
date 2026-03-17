from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc


def rng_color(rng: random.Random) -> Tuple[int, int, int]:
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def write_clip(path: Path, frames: list[np.ndarray], fps: int) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def make_real_frames(rng: random.Random, size: int, num_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    base_color = np.array([220, 220, 230], dtype=np.uint8)
    for i in range(num_frames):
        frame = np.ones((size, size, 3), dtype=np.uint8) * base_color
        # Add low-frequency gradient shift over time
        shift = int(10 * math.sin(i / 6.0))
        frame = np.clip(frame + shift, 0, 255)
        # Add a stable rectangle
        x0, y0 = size // 6, size // 6
        x1, y1 = size - size // 6, size - size // 6
        cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 200), thickness=2)
        frames.append(frame)
    return frames


def make_fake_frames(rng: random.Random, size: int, num_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(num_frames):
        # High-frequency noise background
        frame = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        # Add random moving lines/ellipses
        for _ in range(10):
            color = rng_color(rng)
            x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
            x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
            cv2.line(frame, (x0, y0), (x1, y1), color, rng.randint(1, 3))
            rx = rng.randint(5, size // 3)
            ry = rng.randint(5, size // 3)
            cv2.ellipse(frame, (x0, y0), (rx, ry), 0, 0, 360, color, 2)
        # Sharpen to emphasize artifacts
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)
        frames.append(frame)
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description="Generate tiny synthetic video dataset with real/ and fake/ MP4 clips")
    p.add_argument("--out", type=str, default="training/videos", help="Output dataset root")
    p.add_argument("--num-per-class", type=int, default=10, help="Total videos per class (train+val)")
    p.add_argument("--size", type=int, default=224, help="Frame size (square)")
    p.add_argument("--frames", type=int, default=48, help="Frames per clip")
    p.add_argument("--fps", type=int, default=12, help="Frames per second")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (0-1)")
    p.add_argument("--seed", type=int, default=321, help="Random seed")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for split in ("train", "val"):
        for cls in ("real", "fake"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    n_val = int(math.floor(args.num_per_class * args.val_ratio))
    n_train = args.num_per_class - n_val

    # REAL
    for i in range(n_train):
        frames = make_real_frames(rng, args.size, args.frames)
        write_clip(out / "train" / "real" / f"real_{i:03d}.mp4", frames, args.fps)
    for i in range(n_val):
        frames = make_real_frames(rng, args.size, args.frames)
        write_clip(out / "val" / "real" / f"real_{i:03d}.mp4", frames, args.fps)

    # FAKE
    for i in range(n_train):
        frames = make_fake_frames(rng, args.size, args.frames)
        write_clip(out / "train" / "fake" / f"fake_{i:03d}.mp4", frames, args.fps)
    for i in range(n_val):
        frames = make_fake_frames(rng, args.size, args.frames)
        write_clip(out / "val" / "fake" / f"fake_{i:03d}.mp4", frames, args.fps)

    print(f"Done. Wrote train: {2*n_train} videos, val: {2*n_val} videos to {out}")


if __name__ == "__main__":
    main()


