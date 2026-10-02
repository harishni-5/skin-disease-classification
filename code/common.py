"""
Shared dataset, model, and utility code for the SkinCAW M1-M5 retraining.

This module is deliberately a SINGLE source of truth for the backbone, data
pipeline, and CAW mechanism, so every stage (M1-M5) and every ablation variant
uses identical preprocessing, architecture family, and evaluation code. That
directly addresses the reviewer comment that M1-M5 in the original submission
were not a controlled comparison (different frameworks, inconsistent
normalization, different augmentation implementations, etc).

Key differences from the original notebooks (documented here for the
response-to-reviewers letter):
  1. Every stage now uses PyTorch + the same ResNet-50 backbone (the original
     M1-M4 used TensorFlow/Keras and M5 used PyTorch -- different frameworks
     make the comparison uncontrolled).
  2. Every stage applies the same resize/normalize preprocessing (ImageNet
     mean/std). The original M5 training loop that produced
     final_model_caw_m5.pth did NOT normalize images -- this was a bug.
  3. Gabor energy targets are recomputed with ONE consistent formula for
     every image (the original repo had three different, inconsistent Gabor
     formulas across notebooks, and train/val used different ones).
  4. Gabor targets are looked up by file path, not by dataset index order,
     to eliminate a class of silent misalignment bugs possible with
     index-based lookup + ImageFolder's OS-dependent file ordering.
  5. The ConceptAttentionWhitening module's channel-attention computation is
     fixed: the original implementation's `torch.einsum('bn,bchw->bchw', ...)`
     silently collapses the per-concept similarity vector into a single
     scalar gate (sum over the concept axis), discarding per-concept
     distinction entirely. The corrected version below produces a genuine
     per-channel gate derived from a softmax-weighted blend of the concept
     directions. This is still NOT statistical (covariance) whitening in the
     sense of Chen et al. 2020 -- it is channel attention guided by learned
     concept prototypes -- and the manuscript should describe it as such.
"""
import os
import json
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = [
    "class0_normal", "class1_acne", "class2_wrinkles",
    "class3_Eczema", "class4_Rosacea", "class5_dark_spots",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(augment: bool):
    base = [
        transforms.Resize((224, 224)),
    ]
    if augment:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(base)


class SkinDataset(Dataset):
    """ImageFolder-style dataset that also returns a Gabor energy target,
    looked up by relative file path (robust to ordering)."""

    def __init__(self, root_dir, gabor_targets: dict, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # (path, label_idx)
        self.classes = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            cdir = os.path.join(root_dir, c)
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(cdir, fname), self.class_to_idx[c]))
        self.gabor_targets = gabor_targets

    def __len__(self):
        return len(self.samples)

    def _gabor_for(self, path):
        # keys in gabor_targets.json are relative to the SKIN_CLEAN_DEDUP dir,
        # e.g. "train/class1_acne/Acne_30.jpg"
        rel = os.path.relpath(path, os.path.dirname(self.root_dir.rstrip("/")))
        val = self.gabor_targets.get(rel)
        if val is None:
            # fall back: try matching by basename in case of path differences
            base = os.path.basename(path)
            for k, v in self.gabor_targets.items():
                if k.endswith(base):
                    val = v
                    break
        return float(val) if val is not None else 0.0

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Cast explicitly to float32 here. The gabor targets are loaded from
        # JSON as Python floats, and PyTorch's default_collate infers
        # float64 (double) for plain Python floats -- MPS cannot hold a
        # float64 tensor on-device, so this must be float32 *before* collation.
        gabor = torch.tensor(self._gabor_for(path), dtype=torch.float32)
        return img, label, gabor, path

    def class_counts(self):
        c = Counter(label for _, label in self.samples)
        return {self.classes[i]: c.get(i, 0) for i in range(len(self.classes))}


def load_gabor_targets(dataset_root):
    """dataset_root should be the SKIN_CLEAN_DEDUP directory containing
    gabor_targets.json (produced by compute_gabor.py)."""
    path = os.path.join(dataset_root, "gabor_targets.json")
    with open(path) as f:
        data = json.load(f)
    return data["values"]


def make_sampler(dataset: SkinDataset, oversample: bool):
    if not oversample:
        return None
    counts = Counter(label for _, label in dataset.samples)
    total = len(dataset)
    weight_per_class = {c: total / cnt for c, cnt in counts.items()}
    sample_weights = [weight_per_class[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=total, replacement=True)


def compute_class_weights(dataset: SkinDataset, device):
    counts = Counter(label for _, label in dataset.samples)
    total = len(dataset)
    n_classes = len(dataset.classes)
    weights = torch.zeros(n_classes)
    for c in range(n_classes):
        weights[c] = total / (n_classes * counts.get(c, 1))
    return weights.to(device)


class ConceptAttentionWhitening(nn.Module):
    """Concept-vector channel attention over GroupNorm-normalized features.

    NOTE ON NAMING: this module is referred to as "CAW" throughout the
    manuscript for continuity with the original project name, but it does
    NOT perform statistical (covariance) whitening in the sense of Chen et
    al. 2020 ("Concept whitening for interpretable image recognition").
    It performs (1) per-group feature normalization (GroupNorm) followed by
    (2) soft channel attention, where the attention gate is a
    softmax-weighted blend of learned concept prototype directions, keyed by
    each sample's similarity to those prototypes. The manuscript's Methods
    section should state this precisely and distinguish it from both Chen et
    al. 2020 and the MICCAI 2024 CAW paper cited by the reviewer.
    """

    def __init__(self, in_channels, num_concepts=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.concept_vectors = nn.Parameter(torch.randn(num_concepts, in_channels) * 0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        gap = F.adaptive_avg_pool2d(x_norm, 1).view(B, C)
        gap_n = F.normalize(gap, dim=1)
        concept_n = F.normalize(self.concept_vectors, dim=1)  # [N, C]
        sim = torch.matmul(gap_n, concept_n.T)  # [B, N] cosine similarity to each concept
        attn_weights = F.softmax(sim, dim=1)  # [B, N]
        channel_gate = torch.matmul(attn_weights, concept_n)  # [B, C]
        channel_gate = torch.sigmoid(channel_gate)  # [B, C] in (0, 1)
        x_attended = x_norm * channel_gate.unsqueeze(-1).unsqueeze(-1)
        return x_attended, sim


class ResNet50CAW(nn.Module):
    """One architecture family for all five stages (M1-M5). Components are
    toggled on/off per stage via `use_caw`; the classifier/embed/concept
    heads always exist (for code simplicity) but the training loss decides
    which are actually optimized for a given stage."""

    def __init__(self, num_classes=6, num_concepts=4, embed_dim=128, use_caw=False):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # -> [B, 2048, 7, 7]
        self.use_caw = use_caw
        if use_caw:
            self.caw = ConceptAttentionWhitening(2048, num_concepts)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)
        self.embed_head = nn.Linear(2048, embed_dim)
        self.concept_head = nn.Sequential(nn.Linear(2048, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        feat_map = self.backbone(x)
        concept_sim = None
        if self.use_caw:
            feat_map, concept_sim = self.caw(feat_map)
        pooled = self.pool(feat_map).view(x.size(0), -1)
        logits = self.classifier(pooled)
        embedding = F.normalize(self.embed_head(pooled), dim=1)
        gabor_pred = self.concept_head(pooled).squeeze(-1)
        return {
            "logits": logits,
            "embedding": embedding,
            "gabor_pred": gabor_pred,
            "concept_sim": concept_sim,
        }


# ---- Stage / ablation presets -------------------------------------------------
# Each preset is a dict of flags consumed by train.py. `use_caw` controls the
# attention module; `use_contrastive` / `use_gabor` control which auxiliary
# losses are active. This lets a single script reproduce both the M1-M5
# narrative AND the finer-grained controlled ablation the reviewers asked for
# (isolating CAW-attention, contrastive, and Gabor supervision individually).
STAGE_PRESETS = {
    # --- M1-M5 progression (paper's main narrative, re-run on the clean split) ---
    "m1": dict(augment=False, class_weight=False, oversample=False, use_caw=False, use_contrastive=False, use_gabor=False),
    "m2": dict(augment=True,  class_weight=False, oversample=False, use_caw=False, use_contrastive=False, use_gabor=False),
    "m3": dict(augment=True,  class_weight=True,  oversample=False, use_caw=False, use_contrastive=False, use_gabor=False),
    "m4": dict(augment=True,  class_weight=True,  oversample=True,  use_caw=False, use_contrastive=False, use_gabor=False),
    "m5": dict(augment=True,  class_weight=True,  oversample=True,  use_caw=True,  use_contrastive=True,  use_gabor=True),
    # --- controlled ablation isolating each M5 component (reviewer comment #4) ---
    "m5_caw_only":      dict(augment=True, class_weight=True, oversample=True, use_caw=True,  use_contrastive=False, use_gabor=False),
    "m5_contrast_only": dict(augment=True, class_weight=True, oversample=True, use_caw=False, use_contrastive=True,  use_gabor=False),
    "m5_gabor_only":    dict(augment=True, class_weight=True, oversample=True, use_caw=False, use_contrastive=False, use_gabor=True),
}


def compute_loss(outputs, labels, gabor_targets, class_weights, cfg, lambda_contrast=0.2, lambda_gabor=0.3):
    logits = outputs["logits"]
    if cfg["class_weight"]:
        loss_ce = F.cross_entropy(logits, labels, weight=class_weights)
    else:
        loss_ce = F.cross_entropy(logits, labels)

    loss = loss_ce
    parts = {"ce": loss_ce.item()}

    if cfg["use_contrastive"]:
        embeddings = outputs["embedding"]
        n = embeddings.size(0)
        if n >= 2:
            m = n - (n % 2)
            anchor = embeddings[:m:2]
            positive = embeddings[1:m:2]
            lbl = labels[:m]
            target = (lbl[::2] == lbl[1::2]).float() * 2 - 1
            loss_contrast = F.cosine_embedding_loss(anchor, positive, target)
        else:
            loss_contrast = torch.tensor(0.0, device=logits.device)
        loss = loss + lambda_contrast * loss_contrast
        parts["contrast"] = loss_contrast.item()

    if cfg["use_gabor"]:
        loss_gabor = F.mse_loss(outputs["gabor_pred"], gabor_targets)
        loss = loss + lambda_gabor * loss_gabor
        parts["gabor"] = loss_gabor.item()

    return loss, parts
