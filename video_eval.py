from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.metrics import accuracy_score

from .video_model import VideoModelConfig, CnnLstmVideoClassifier
from .video_train import list_videos, sample_frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="models/deepfake_video_cnnlstm.pth")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    items = list_videos(Path(args.data_dir) / "val")
    ckpt = torch.load(args.model, map_location="cpu")
    cfg_dict = ckpt.get("video_model_cfg", {})
    cfg = VideoModelConfig(**cfg_dict) if cfg_dict else VideoModelConfig()
    model = CnnLstmVideoClassifier(cfg)
    model.load_state_dict(ckpt["model_state"]) 
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for path, lbl in items:
            clip = sample_frames(path, num_frames=args.frames, size=args.image_size).unsqueeze(0)
            logits = model(clip)
            pred = int(logits.argmax(1).item())
            y_true.append(lbl)
            y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    print(f"VIDEO_VAL_ACC={acc:.4f}")


if __name__ == "__main__":
    main()






