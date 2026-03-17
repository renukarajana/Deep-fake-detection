from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score

from .data import DataConfig, build_dataloaders
from .model import ModelConfig, build_model


def main() -> None:
    ckpt_path = Path("models/deepfake_cnn.pth")
    if not ckpt_path.exists():
        print("ERROR: Checkpoint not found at models/deepfake_cnn.pth")
        return

    data_cfg = DataConfig(data_dir="training/data", image_size=224, batch_size=64, num_workers=0)
    train_loader, val_loader = build_dataloaders(data_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg_dict = ckpt.get("model_cfg", {})
    model_cfg = ModelConfig(**model_cfg_dict) if model_cfg_dict else ModelConfig()
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Train accuracy
    y_true_tr, y_pred_tr = [], []
    with torch.no_grad():
        for images, labels in train_loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true_tr.extend(labels.tolist())
            y_pred_tr.extend(preds.tolist())
    train_acc = accuracy_score(y_true_tr, y_pred_tr) if y_true_tr else 0.0

    # Val accuracy
    y_true_va, y_pred_va = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true_va.extend(labels.tolist())
            y_pred_va.extend(preds.tolist())
    val_acc = accuracy_score(y_true_va, y_pred_va) if y_true_va else 0.0

    print(f"TRAIN_ACC={train_acc:.4f}")
    print(f"VAL_ACC={val_acc:.4f}")


if __name__ == "__main__":
    main()


