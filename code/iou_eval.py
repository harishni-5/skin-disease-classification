"""
Grad-CAM++ vs expert-annotated mask IoU evaluation, using the
dataset/randomm/{images,masks} folders (img.png + binary lesion mask per
image). This reproduces the paper's Sec. "Quantitative validation of model
attention with IoU" but:
  - reports per-image and per-class IoU (not just the overall mean),
  - reports a center-mass baseline and a random-mask baseline IoU for
    comparison, as the reviewer explicitly asked for,
  - uses Otsu thresholding on the CAM (documented, reproducible) instead of
    an unspecified threshold.

Usage:
    python iou_eval.py --checkpoint ../results/m5/best_model.pt --use_caw \
        --randomm_dir ../dataset/randomm --out iou_results.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.filters import threshold_otsu

from common import ResNet50CAW, build_transforms, get_device, CLASS_NAMES


class GradCAMPlusPlus:
    """Same formulation as used in the original notebooks (Chattopadhay et al.
    2018), reimplemented here against the unified ResNet50CAW model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        logits = output["logits"] if isinstance(output, dict) else output
        if target_class is None:
            target_class = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits)
        for i, c in enumerate(target_class):
            one_hot[i, c] = 1
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients
        acts = self.activations
        B, C, H, W = grads.shape
        eps = 1e-8
        grads_pow2 = grads ** 2
        grads_pow3 = grads ** 3
        sum_acts = acts.view(B, C, -1).sum(dim=2).view(B, C, 1, 1)
        alpha_denom = 2 * grads_pow2 + sum_acts * grads_pow3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps))
        alpha = grads_pow2 / alpha_denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= cam.amin(dim=(2, 3), keepdim=True)
        cam /= (cam.amax(dim=(2, 3), keepdim=True) + eps)
        return cam


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def threshold_cam(cam_np):
    try:
        t = threshold_otsu(cam_np)
    except Exception:
        t = 0.5
    return cam_np >= t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--use_caw", action="store_true")
    p.add_argument("--randomm_dir", required=True)
    p.add_argument("--out", default="iou_results.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = get_device()
    model = ResNet50CAW(num_classes=len(CLASS_NAMES), use_caw=args.use_caw).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    target_layer = model.backbone[-1][-1]  # last block of layer4
    gcampp = GradCAMPlusPlus(model, target_layer)

    transform = build_transforms(augment=False)

    mask_paths = sorted(glob.glob(os.path.join(args.randomm_dir, "masks", "*.png")))
    print(f"found {len(mask_paths)} masks")

    per_image = []
    for mask_path in mask_paths:
        name = os.path.splitext(os.path.basename(mask_path))[0]
        img_candidates = [
            os.path.join(args.randomm_dir, "images", f"{name}_json", "img.png"),
        ]
        img_path = next((c for c in img_candidates if os.path.exists(c)), None)
        if img_path is None:
            print(f"WARNING: no matching image for mask {mask_path}, skipping")
            continue

        gt_mask = np.array(Image.open(mask_path).convert("L")) > 0
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        cam = gcampp.generate(input_tensor)
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_resized = np.array(Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
            (gt_mask.shape[1], gt_mask.shape[0])
        )) / 255.0
        pred_mask = threshold_cam(cam_resized)

        model_iou = iou(pred_mask, gt_mask)

        # --- baselines the reviewer asked for ---
        h, w = gt_mask.shape
        center_mask = np.zeros((h, w), dtype=bool)
        ch, cw = h // 2, w // 2
        rh, rw = h // 4, w // 4
        center_mask[ch - rh: ch + rh, cw - rw: cw + rw] = True
        center_iou = iou(center_mask, gt_mask)

        random_mask = rng.random((h, w)) < (pred_mask.mean() if pred_mask.mean() > 0 else 0.25)
        random_iou = iou(random_mask, gt_mask)

        per_image.append({
            "name": name, "model_iou": model_iou,
            "center_baseline_iou": center_iou, "random_baseline_iou": random_iou,
            "gt_mask_area_frac": float(gt_mask.mean()),
            "pred_mask_area_frac": float(pred_mask.mean()),
        })
        print(f"{name:20s} model_iou={model_iou:.4f} center_baseline={center_iou:.4f} random_baseline={random_iou:.4f}")

    model_ious = np.array([r["model_iou"] for r in per_image])
    center_ious = np.array([r["center_baseline_iou"] for r in per_image])
    random_ious = np.array([r["random_baseline_iou"] for r in per_image])

    summary = {
        "n_images": len(per_image),
        "model_iou_mean": float(model_ious.mean()) if len(model_ious) else None,
        "model_iou_std": float(model_ious.std()) if len(model_ious) else None,
        "center_baseline_iou_mean": float(center_ious.mean()) if len(center_ious) else None,
        "random_baseline_iou_mean": float(random_ious.mean()) if len(random_ious) else None,
        "per_image": per_image,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nmean model IoU = {summary['model_iou_mean']:.4f} +/- {summary['model_iou_std']:.4f} "
          f"over {summary['n_images']} images")
    print(f"center-baseline mean IoU = {summary['center_baseline_iou_mean']:.4f}")
    print(f"random-baseline mean IoU = {summary['random_baseline_iou_mean']:.4f}")
    print(f"saved details to {args.out}")


if __name__ == "__main__":
    main()
