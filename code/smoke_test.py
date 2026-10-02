"""
Run this FIRST, before any full training. It runs one tiny batch through the
whole pipeline (data loading, model forward, loss, backward) for every stage
and prints PASS/FAIL. Takes well under a minute. Catches path/shape/typo bugs
early instead of an hour into a real training run.

Usage (from the retrain/ directory):
    python smoke_test.py --data_root ../dataset/SKIN_CLEAN_DEDUP
"""
import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from common import (
    SkinDataset, load_gabor_targets, build_transforms, compute_class_weights,
    ResNet50CAW, STAGE_PRESETS, compute_loss, set_seed, get_device, CLASS_NAMES,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    args = p.parse_args()

    set_seed(0)
    device = get_device()
    print(f"device = {device}")

    gabor_targets = load_gabor_targets(args.data_root)
    transform = build_transforms(augment=False)
    train_ds = SkinDataset(f"{args.data_root}/train", gabor_targets, transform=transform)
    print(f"dataset loaded: {len(train_ds)} train images, classes={train_ds.classes}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    images, labels, gabor, paths = next(iter(loader))
    print(f"batch shapes: images={tuple(images.shape)} labels={tuple(labels.shape)} gabor={tuple(gabor.shape)}")

    class_weights = compute_class_weights(train_ds, device)

    for stage, cfg in STAGE_PRESETS.items():
        try:
            model = ResNet50CAW(num_classes=len(CLASS_NAMES), use_caw=cfg["use_caw"]).to(device)
            optimizer = Adam(model.parameters(), lr=1e-4)
            outputs = model(images.to(device))
            loss, parts = compute_loss(
                outputs, labels.to(device), gabor.float().to(device), class_weights, cfg
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[PASS] stage={stage:20s} loss={loss.item():.4f} parts={parts}")
        except Exception as e:
            print(f"[FAIL] stage={stage:20s} error={e}")
            raise

    print("\nAll stages passed a forward+backward step. Safe to start real training.")


if __name__ == "__main__":
    main()
