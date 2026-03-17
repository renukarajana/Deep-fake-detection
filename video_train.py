from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import transforms

from .video_model import VideoModelConfig, CnnLstmVideoClassifier


def sample_frames(video_path: Path, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = [int(i * total / num_frames) for i in range(num_frames)]
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(tfm(frame_rgb))
    cap.release()
    if not frames:
        # fallback black frames
        frames = [torch.zeros(3, size, size) for _ in range(num_frames)]
    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    clip = torch.stack(frames[:num_frames], dim=0)  # [T, 3, H, W]
    return clip


def list_videos(root: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    class_to_idx = {"real": 0, "fake": 1}
    for cls, idx in class_to_idx.items():
        for p in (root / cls).rglob("*.mp4"):
            items.append((p, idx))
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True, help="Root with train/ and val/ subfolders of videos")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out-model", type=str, default="models/deepfake_video_cnnlstm.pth")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_items = list_videos(Path(args.data_dir) / "train")
    val_items = list_videos(Path(args.data_dir) / "val")

    model_cfg = VideoModelConfig()
    model = CnnLstmVideoClassifier(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(items: List[Tuple[Path, int]], train: bool) -> Tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        y_true: List[int] = []
        y_pred: List[int] = []
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            clips, labels = [], []
            for path, lbl in batch:
                clip = sample_frames(path, num_frames=args.frames, size=args.image_size)
                clips.append(clip)
                labels.append(lbl)
            if not clips:
                continue
            x = torch.stack(clips, dim=0).to(device)  # [B, T, 3, H, W]
            y = torch.tensor(labels, dtype=torch.long).to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = criterion(logits, y)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())
        avg_loss = total_loss / max(1, len(items))
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        return avg_loss, acc

    best_val = 0.0
    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_items, train=True)
        va_loss, va_acc = run_epoch(val_items, train=False)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "video_model_cfg": model_cfg.__dict__,
            }, out_path)
            print(f"Saved best video model to {out_path}")


if __name__ == "__main__":
    main()






