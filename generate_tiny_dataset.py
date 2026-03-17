from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def rng_color(rng: random.Random) -> Tuple[int, int, int]:
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def make_real_like(rng: random.Random, size: int) -> Image.Image:
    # Smooth gradients + simple geometric features to simulate "structured" content
    base = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(base)
    # Add soft radial gradient
    cx, cy = size // 2, size // 2
    for r in range(min(cx, cy), 0, -4):
        shade = 255 - int(120 * (r / (size / 2)))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(shade, shade, shade))
    # Add a few stable shapes
    for _ in range(3):
        color = rng_color(rng)
        x0, y0 = rng.randint(0, size // 2), rng.randint(0, size // 2)
        x1, y1 = x0 + rng.randint(size // 4, size - x0), y0 + rng.randint(size // 4, size - y0)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
    return base.filter(ImageFilter.GaussianBlur(radius=1.5))


def make_fake_like(rng: random.Random, size: int) -> Image.Image:
    # Higher-frequency noise + distortions to simulate "artifacty" content
    arr = (rng.random() * 255 * np.ones((size, size, 3))).astype(np.uint8)
    # Mix in perlin-like coarse noise by downsampling and upsampling
    small = Image.fromarray(arr).resize((max(4, size // 8), max(4, size // 8)), Image.BILINEAR)
    up = small.resize((size, size), Image.NEAREST)
    img = Image.blend(Image.fromarray(arr), up, 0.5)
    draw = ImageDraw.Draw(img)
    # Add random lines/ellipses
    for _ in range(8):
        color = rng_color(rng)
        x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        draw.line((x0, y0, x1, y1), fill=color, width=rng.randint(1, 3))
        rx = rng.randint(5, size // 3)
        ry = rng.randint(5, size // 3)
        draw.ellipse((x0, y0, min(size - 1, x0 + rx), min(size - 1, y0 + ry)), outline=color, width=2)
    return img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=180, threshold=3))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a tiny synthetic dataset with real/ and fake/ for quick tests")
    p.add_argument("--out", type=str, default="training/data", help="Output dataset root")
    p.add_argument("--num-per-class", type=int, default=50, help="Total images per class (train+val)")
    p.add_argument("--image-size", type=int, default=224, help="Square image size")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (0-1)")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "train" / "real").mkdir(parents=True, exist_ok=True)
    (out / "train" / "fake").mkdir(parents=True, exist_ok=True)
    (out / "val" / "real").mkdir(parents=True, exist_ok=True)
    (out / "val" / "fake").mkdir(parents=True, exist_ok=True)

    n_val = int(math.floor(args.num_per_class * args.val_ratio))
    n_train = args.num_per_class - n_val

    # Generate REAL
    for i in range(n_train):
        img = make_real_like(rng, args.image_size)
        img.save(out / "train" / "real" / f"real_{i:04d}.jpg", quality=95)
    for i in range(n_val):
        img = make_real_like(rng, args.image_size)
        img.save(out / "val" / "real" / f"real_{i:04d}.jpg", quality=95)

    # Generate FAKE
    for i in range(n_train):
        img = make_fake_like(rng, args.image_size)
        img.save(out / "train" / "fake" / f"fake_{i:04d}.jpg", quality=95)
    for i in range(n_val):
        img = make_fake_like(rng, args.image_size)
        img.save(out / "val" / "fake" / f"fake_{i:04d}.jpg", quality=95)

    print(f"Done. Wrote train: {2*n_train} images, val: {2*n_val} images to {out}")


if __name__ == "__main__":
    main()


