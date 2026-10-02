"""
Unified training script for stages M1-M5 (and the extra ablation-isolation
presets). Same backbone, same optimizer, same preprocessing, same split for
every stage -- only the flags in STAGE_PRESETS differ. This directly answers
the reviewer's demand for a controlled ablation using one pipeline.

Usage (run from the `retrain/` directory):

    python train.py --stage m1 --data_root ../dataset/SKIN_CLEAN_DEDUP --out_dir ../results/m1
    python train.py --stage m2 --data_root ../dataset/SKIN_CLEAN_DEDUP --out_dir ../results/m2
    ... etc for m3, m4, m5, and optionally m5_caw_only / m5_contrast_only / m5_gabor_only

Each run writes to <out_dir>/:
    history.json        per-epoch train/val loss components + val accuracy
    best_model.pt        best checkpoint (by val loss)
    final_model.pt        last-epoch checkpoint
    val_predictions.csv    per-image prediction on the val set (best checkpoint)
    metrics.json        final accuracy / macro & weighted F1 / per-class P,R,F1 / confusion matrix
    run_config.json        exact hyperparameters used (for reproducibility / the paper's methods section)
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)

from common import (
    SkinDataset, load_gabor_targets, build_transforms, make_sampler,
    compute_class_weights, ResNet50CAW, STAGE_PRESETS, compute_loss,
    set_seed, get_device, CLASS_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=list(STAGE_PRESETS.keys()))
    p.add_argument("--data_root", required=True, help="path to SKIN_CLEAN_DEDUP")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lambda_contrast", type=float, default=0.2)
    p.add_argument("--lambda_gabor", type=float, default=0.3)
    p.add_argument("--max_batches_per_epoch", type=int, default=0,
                    help="debug/smoke-test only: cap batches per epoch (0 = no cap)")
    return p.parse_args()


def run_epoch(model, loader, optimizer, class_weights, cfg, args, device, train: bool):
    model.train(mode=train)
    totals = {"ce": 0.0, "contrast": 0.0, "gabor": 0.0, "n": 0}
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for bi, (images, labels, gabor, paths) in enumerate(loader):
            if args.max_batches_per_epoch and bi >= args.max_batches_per_epoch:
                break
            images = images.to(device)
            labels = labels.to(device)
            gabor = gabor.float().to(device)

            outputs = model(images)
            loss, parts = compute_loss(
                outputs, labels, gabor, class_weights, cfg,
                lambda_contrast=args.lambda_contrast, lambda_gabor=args.lambda_gabor,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = images.size(0)
            totals["ce"] += parts.get("ce", 0.0) * bs
            totals["contrast"] += parts.get("contrast", 0.0) * bs
            totals["gabor"] += parts.get("gabor", 0.0) * bs
            totals["n"] += bs

            preds = outputs["logits"].argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    n = max(totals["n"], 1)
    avg = {k: totals[k] / n for k in ("ce", "contrast", "gabor")}
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg, acc


def evaluate_and_save(model, loader, device, out_dir, cfg):
    model.eval()
    all_preds, all_labels, all_paths, all_probs = [], [], [], []
    with torch.no_grad():
        for images, labels, gabor, paths in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
            all_probs.extend(probs.tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() > 0 else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES)))).tolist()
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm,
        "confusion_matrix_labels": CLASS_NAMES,
        "n_correct": int(sum(p == l for p, l in zip(all_preds, all_labels))),
        "n_total": len(all_labels),
        "sklearn_classification_report": report,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "val_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "pred_label", "correct"] + [f"prob_{c}" for c in CLASS_NAMES])
        for path, t, p, probs in zip(all_paths, all_labels, all_preds, all_probs):
            w.writerow([path, CLASS_NAMES[t], CLASS_NAMES[p], int(t == p)] + probs)

    return metrics


def main():
    args = parse_args()
    cfg = STAGE_PRESETS[args.stage]
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"[{args.stage}] device={device} cfg={cfg}")

    gabor_targets = load_gabor_targets(args.data_root)
    train_transform = build_transforms(augment=cfg["augment"])
    val_transform = build_transforms(augment=False)

    train_ds = SkinDataset(os.path.join(args.data_root, "train"), gabor_targets, transform=train_transform)
    val_ds = SkinDataset(os.path.join(args.data_root, "val"), gabor_targets, transform=val_transform)
    print(f"train={len(train_ds)} val={len(val_ds)} classes={train_ds.class_counts()}")

    sampler = make_sampler(train_ds, oversample=cfg["oversample"])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_weights = compute_class_weights(train_ds, device) if cfg["class_weight"] else None

    model = ResNet50CAW(num_classes=len(CLASS_NAMES), use_caw=cfg["use_caw"]).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    epochs_without_improve = 0
    start_epoch = 1

    # ---- resume support: if this stage was interrupted (Ctrl+C, terminal
    # closed, laptop slept, etc.), pick up from the last completed epoch
    # instead of starting over. Just re-run the exact same command. ----
    ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt["best_val_loss"]
        epochs_without_improve = ckpt["epochs_without_improve"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[{args.stage}] RESUMING from checkpoint at epoch {ckpt['epoch']} "
              f"(best_val_loss={best_val_loss:.4f}, {epochs_without_improve} epochs without improvement)")
        if epochs_without_improve >= args.patience:
            print(f"[{args.stage}] checkpoint shows early-stopping condition already met "
                  f"({epochs_without_improve} >= patience {args.patience}); skipping straight "
                  f"to final evaluation instead of re-entering the training loop.")
            start_epoch = args.epochs + 1
        elif start_epoch > args.epochs:
            print(f"[{args.stage}] checkpoint already reached epoch {ckpt['epoch']} "
                  f">= --epochs {args.epochs}; skipping straight to final evaluation.")

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump({"stage": args.stage, "cfg": cfg, "args": vars(args)}, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_avg, train_acc = run_epoch(model, train_loader, optimizer, class_weights, cfg, args, device, train=True)
        val_avg, val_acc = run_epoch(model, val_loader, optimizer, class_weights, cfg, args, device, train=False)

        val_loss = val_avg["ce"] + args.lambda_contrast * val_avg["contrast"] + args.lambda_gabor * val_avg["gabor"]
        dt = time.time() - t0

        row = {
            "epoch": epoch, "train": train_avg, "train_acc": train_acc,
            "val": val_avg, "val_acc": val_acc, "val_loss": val_loss, "seconds": dt,
        }
        history.append(row)
        print(f"epoch {epoch}/{args.epochs} | train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"val_loss={val_loss:.4f} ({dt:.1f}s)")

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
        else:
            epochs_without_improve += 1

        # Save a full resumable checkpoint after EVERY epoch (not just on
        # improvement) so an interrupted run can pick back up here. This is
        # deliberately separate from best_model.pt, which stores weights
        # only and is what gets evaluated/shipped.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improve": epochs_without_improve,
            "history": history,
        }, ckpt_path)

        if epochs_without_improve >= args.patience:
            print(f"early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))

    # final evaluation using the BEST checkpoint (matches how the headline
    # number should be reported: model selected on val loss, then scored)
    best_path = os.path.join(args.out_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    metrics = evaluate_and_save(model, val_loader, device, args.out_dir, cfg)
    print(f"[{args.stage}] FINAL val accuracy={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} ({metrics['n_correct']}/{metrics['n_total']})")


if __name__ == "__main__":
    main()
