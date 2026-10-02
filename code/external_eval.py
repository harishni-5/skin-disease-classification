"""
Runs the trained model on the external "unseen" image set
(dataset/randomm/images/*/img.png) and scores it automatically instead of by
eyeballing printed predictions, addressing the reviewer's ask for the
external-test result to be reported with proper uncertainty.

Ground truth is inferred from the folder/file name (e.g. "acne1" -> class1_acne,
"eczema" -> class3_Eczema, "rosacea" -> class4_Rosacea, "normal"/"nor" ->
class0_normal, "wrinkle" -> class2_wrinkles, "dark_sp"/"darkk" -> class5_dark_spots).
Double-check the printed mapping against the actual images before trusting the
result -- this is a heuristic, not an authoritative label file, and that
itself is worth stating plainly in the manuscript (reviewer comment: labels,
sources, and selection criteria for the external set were not reported).

Usage:
    python external_eval.py --checkpoint ../results/m5/best_model.pt \
        --images_dir ../dataset/randomm/images --use_caw
"""
import argparse
import json
import os

import torch
from PIL import Image

from common import ResNet50CAW, build_transforms, get_device, CLASS_NAMES

NAME_TO_CLASS = [
    (("acne",), "class1_acne"),
    (("eczema",), "class3_Eczema"),
    (("rosacea",), "class4_Rosacea"),
    (("wrinkle",), "class2_wrinkles"),
    (("dark_sp", "darkk", "dark_Sp"), "class5_dark_spots"),
    (("normal", "nor", "norr"), "class0_normal"),
]


def infer_true_class(name: str):
    lname = name.lower()
    for keys, cls in NAME_TO_CLASS:
        if any(k.lower() in lname for k in keys):
            return cls
    return None


def find_images(images_dir):
    """Handles both a flat folder of images and the labelme-style
    <name>_json/img.png subfolder layout used in dataset/randomm/images."""
    found = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower() in ("img.png",) or f.lower().endswith((".jpg", ".jpeg", ".png")):
                if f.lower() in ("label.png", "label_viz.png"):
                    continue
                folder_name = os.path.basename(root) if f.lower() == "img.png" else os.path.splitext(f)[0]
                found.append((os.path.join(root, f), folder_name))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--use_caw", action="store_true")
    p.add_argument("--out", default="external_eval_results.json")
    args = p.parse_args()

    device = get_device()
    model = ResNet50CAW(num_classes=len(CLASS_NAMES), use_caw=args.use_caw).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = build_transforms(augment=False)
    images = find_images(args.images_dir)
    print(f"found {len(images)} candidate images under {args.images_dir}")

    results = []
    n_correct = 0
    n_labeled = 0
    for path, name in sorted(images):
        true_cls = infer_true_class(name)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            pred_idx = out["logits"].argmax(dim=1).item()
        pred_cls = CLASS_NAMES[pred_idx]
        correct = (pred_cls == true_cls) if true_cls else None
        if true_cls is not None:
            n_labeled += 1
            n_correct += int(correct)
        results.append({
            "path": path, "name": name, "inferred_true_class": true_cls,
            "predicted_class": pred_cls, "correct": correct,
        })
        print(f"{name:40s} true={str(true_cls):20s} pred={pred_cls:20s} correct={correct}")

    summary = {
        "n_total_images": len(images),
        "n_with_inferred_label": n_labeled,
        "n_correct": n_correct,
        "accuracy": n_correct / n_labeled if n_labeled else None,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{n_correct}/{n_labeled} correct on images with an inferred label. "
          f"Saved details to {args.out}. Run stats.py binomial_ci {n_correct} {n_labeled} "
          f"for the confidence interval reviewers asked for.")


if __name__ == "__main__":
    main()
