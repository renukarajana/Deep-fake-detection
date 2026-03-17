from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image
import shutil


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
    return files


def split_items(items: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    items = items.copy()
    rng.shuffle(items)
    n_val = int(math.floor(len(items) * val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items


def save_image(src: Path, dst: Path, resize: int | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if resize is None:
        shutil.copy2(src, dst)
        return
    img = Image.open(src).convert("RGB").resize((resize, resize))
    # Preserve extension when reasonable, default to JPEG
    ext = dst.suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
        dst = dst.with_suffix(ext)
    format_map = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".webp": "WEBP"}
    img.save(dst, format=format_map.get(ext, "JPEG"), quality=95)


def process_class(class_name: str, src_dir: Path, out_dir: Path, val_ratio: float, resize: int | None, seed: int) -> Tuple[int, int]:
    src_class = src_dir / class_name
    if not src_class.exists():
        raise FileNotFoundError(f"Expected folder not found: {src_class}")

    items = list_images(src_class)
    train_items, val_items = split_items(items, val_ratio, seed)

    for item in train_items:
        rel = item.relative_to(src_class)
        dst = out_dir / "train" / class_name / rel
        save_image(item, dst, resize)

    for item in val_items:
        rel = item.relative_to(src_class)
        dst = out_dir / "val" / class_name / rel
        save_image(item, dst, resize)

    return len(train_items), len(val_items)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare deepfake dataset: split into train/val and optional resize.")
    p.add_argument("--source", type=str, required=True, help="Path with subfolders 'real' and 'fake'")
    p.add_argument("--out", type=str, required=True, help="Output dataset root (will create train/ and val/)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (0-1)")
    p.add_argument("--resize", type=int, default=None, help="Optional square resize (e.g., 224)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--real-name", type=str, default="real", help="Folder name for real class inside --source")
    p.add_argument("--fake-name", type=str, default="fake", help="Folder name for fake class inside --source")
    args = p.parse_args()

    src_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [args.real_name, args.fake_name]
    total_train = total_val = 0
    for cls in classes:
        ntr, nva = process_class(cls, src_dir, out_dir, args.val_ratio, args.resize, args.seed)
        print(f"Class {cls}: train={ntr} val={nva}")
        total_train += ntr
        total_val += nva

    print(f"Done. Wrote train={total_train}, val={total_val} images to {out_dir}")


if __name__ == "__main__":
    main()


