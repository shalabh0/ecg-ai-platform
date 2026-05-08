# training/train_classifier.py
# MobileNetV3-Small fine-tune on rendered 12-lead ECG PNGs
# 48 images → prototype classifier → saves .pt + .onnx
#
# Usage:
#   python training/train_classifier.py \
#     --data   /kaggle/working/rendered \
#     --output /kaggle/working/models

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# ─── Config ───────────────────────────────────────────────────────────────────

IMG_SIZE   = 224          # MobileNetV3 input
BATCH_SIZE = 8            # small dataset → small batch
EPOCHS     = 40
LR         = 1e-3
LR_HEAD    = 1e-3         # classifier head
LR_BODY    = 1e-5         # unfrozen backbone layers
PATIENCE   = 8            # early stopping
SEED       = 42

CLASSES    = ["abnormal", "normal"]   # torchvision sorts alphabetically
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Train: light augmentation only — dataset too small for heavy transforms.
    Val  : centre crop only.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ─── Dataset split ────────────────────────────────────────────────────────────

def split_dataset(data_dir: str, val_frac: float = 0.2, seed: int = SEED):
    """
    Stratified 80/20 split from ImageFolder.
    Returns (train_dataset, val_dataset, class_weights_for_sampler).
    """
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    full = datasets.ImageFolder(data_dir, transform=get_transforms("train"))
    targets = np.array(full.targets)

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_frac,
        stratify=targets,
        random_state=seed,
    )

    train_ds = Subset(full, train_idx)
    # Val set uses val transforms — need a separate ImageFolder instance
    val_full = datasets.ImageFolder(data_dir, transform=get_transforms("val"))
    val_ds   = Subset(val_full, val_idx)

    # Class weights for sampler (handles any residual imbalance)
    train_targets = targets[train_idx]
    class_counts  = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_targets]

    print(f"[data] Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"[data] Classes: {full.classes}  Counts: {class_counts.tolist()}")
    return train_ds, val_ds, sample_weights, full.classes


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2) -> nn.Module:
    """
    MobileNetV3-Small pretrained on ImageNet.
    Strategy:
      - Freeze entire backbone
      - Replace classifier head (Linear → Dropout → Linear)
      - Unfreeze last 2 InvertedResidual blocks for fine-tuning
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last 2 blocks of features (blocks 10 and 11)
    for block in list(model.features.children())[-2:]:
        for p in block.parameters():
            p.requires_grad = True

    # Replace classifier head
    in_features = model.classifier[0].in_features   # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes),
    )

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total params: {total:,}  Trainable: {trainable:,} "
          f"({100*trainable/total:.1f}%)")
    return model


# ─── Training loop ────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, output_dir: Path, epochs: int = EPOCHS):
    """
    Train with:
      - Differential LR: head gets LR_HEAD, backbone gets LR_BODY
      - CosineAnnealingLR scheduler
      - Early stopping on val accuracy
      - Saves best checkpoint as ecg_classifier.pt
    """
    # Separate param groups for differential LR
    head_params    = list(model.classifier.parameters())
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and
                       not any(p is hp for hp in head_params)]

    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": LR_HEAD},
        {"params": backbone_params, "lr": LR_BODY},
    ], weight_decay=1e-4)

    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    best_val_acc   = 0.0
    patience_count = 0
    history        = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train phase ───────────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()

        scheduler.step()

        # ── Val phase ─────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits  = model(imgs)
                loss    = criterion(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()

        # ── Metrics ───────────────────────────────────────────────────────
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        t_loss  = train_loss / n_train
        t_acc   = train_correct / n_train
        v_loss  = val_loss / n_val
        v_acc   = val_correct / n_val

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train loss {t_loss:.4f} acc {t_acc:.3f} | "
              f"val loss {v_loss:.4f} acc {v_acc:.3f} | "
              f"{elapsed:.1f}s")

        # ── Checkpoint ────────────────────────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            ckpt_path    = output_dir / "ecg_classifier.pt"
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_acc":    best_val_acc,
                "classes":    CLASSES,
            }, ckpt_path)
            print(f"           ✅ Saved best model (val_acc={best_val_acc:.3f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[early stop] No improvement for {PATIENCE} epochs. Stopping.")
                break

    print(f"\n[train] Best val accuracy: {best_val_acc:.3f}")
    return history


# ─── ONNX export ──────────────────────────────────────────────────────────────

def export_onnx(model, output_dir: Path):
    """Export best checkpoint to ONNX for FastAPI inference."""
    ckpt_path = output_dir / "ecg_classifier.pt"
    onnx_path = output_dir / "ecg_classifier.onnx"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval().cpu()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"[onnx] Exported → {onnx_path}")
    return onnx_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser(description="Train ECG classifier")
    p.add_argument("--data",   default="datasets/rendered",
                   help="Root dir with normal/ and abnormal/ subfolders")
    p.add_argument("--output", default="models",
                   help="Where to save .pt and .onnx")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--no-onnx", action="store_true",
                   help="Skip ONNX export")
    return p.parse_args()


if __name__ == "__main__":
    args   = _args()
    output = Path(args.output)

    print(f"[config] Device: {DEVICE}")
    print(f"[config] Data:   {args.data}")
    print(f"[config] Output: {output}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, sample_weights, classes = split_dataset(args.data)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes=len(classes)).to(DEVICE)

    # ── Train ─────────────────────────────────────────────────────────────
    history = train(model, train_loader, val_loader, output, epochs=args.epochs)

    # ── ONNX export ───────────────────────────────────────────────────────
    if not args.no_onnx:
        export_onnx(model, output)

    print("\n[done] Training complete.")
    print(f"  Model : {output}/ecg_classifier.pt")
    print(f"  ONNX  : {output}/ecg_classifier.onnx")