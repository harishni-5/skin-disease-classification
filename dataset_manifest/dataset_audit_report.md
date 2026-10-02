# Dataset audit: deduplication and leakage-safe re-split

This documents exactly how `dataset/SKIN_CLEAN_DEDUP/` was built from
`dataset/SKIN_PROJECT_CLEANED2_edgemix/` (the dataset version matching the
manuscript's Table 1). Use this for the manuscript's Methods/Dataset section
and the response-to-reviewers letter (major comments #1 and #2 in the review
document; comment #5 in the PDF review).

## What was found in the original split

The original train (7,879 images) and val (943 images) folders were pooled
and hashed. Exact byte-for-byte duplicates were found using MD5; near-
duplicates (resized, re-compressed, or lightly-augmented copies of the same
photo) were found using perceptual hashing (`imagehash.phash`, 64-bit,
Hamming distance ≤ 4 — validated against actual pixel correlation; sampled
pairs at the maximum distance still had normalized cross-correlation ≥ 0.90,
confirming these are true near-duplicates and not hash collisions).

- **92 validation images (9.8%)** were byte-identical to a training image
  (same photo, saved under a different filename/extension — e.g.
  `val/eczema-hand-166.jpeg` and `train/t-eczema-hand-166.jpg`).
- **135 validation images (14.3%)** had an exact-or-near-duplicate match in
  training when near-duplicates are included (worst-affected classes:
  Wrinkles 24.0%, Dark Spots 23.0%, Eczema 20.9%).
- Pooling train+val together (8,822 images total) and clustering all
  exact/near duplicates (not just across the train/val boundary, but within
  each split too) found **1,007 duplicate clusters covering 2,205 images
  (25.0% of the pooled dataset)**. The Eczema class was the most affected:
  of 3,132 pooled Eczema images, only 2,183 were visually unique (a 30.3%
  duplication rate) — meaning Eczema's apparent ~3x size advantage over the
  other five classes in the original dataset was partly an artifact of
  duplicate images, not genuinely distinct examples.
- **No cross-class duplicate clusters were found** — i.e., no evidence that
  the same photo was labeled as two different conditions in different
  places. This is worth stating explicitly as a positive data-quality
  finding.

## Rebuild procedure

1. Pool all 8,822 images from the original train+val folders.
2. Union-find clustering: two images are merged into the same cluster if
   they are MD5-identical OR have a perceptual-hash Hamming distance ≤ 4.
3. Keep exactly one representative image per cluster (7,624 unique images
   survive from the original 8,822 — a 13.6% reduction).
4. Re-split by class with a fixed seed (42), 85%/15% stratified train/val,
   **ignoring the original train/val assignment entirely** (so a duplicate
   that used to straddle both splits now has its single surviving copy
   assigned once, to exactly one split).
5. Verified programmatically: zero exact or near-duplicate images between
   the new train and val folders (re-ran the same MD5 + phash check that
   found the original leakage; result was 0/0).

## Before → after counts

| Class | Pooled (orig.) | Unique after dedup | Removed | New train | New val |
|---|---|---|---|---|---|
| Normal | 1,240 | 1,181 | 59 (4.8%) | 1,004 | 177 |
| Acne | 1,185 | 1,133 | 52 (4.4%) | 963 | 170 |
| Wrinkles | 1,100 | 1,029 | 71 (6.5%) | 875 | 154 |
| Eczema | 3,132 | 2,183 | 949 (30.3%) | 1,856 | 327 |
| Rosacea | 1,108 | 1,091 | 17 (1.5%) | 927 | 164 |
| Dark Spots | 1,057 | 1,007 | 50 (4.7%) | 856 | 151 |
| **Total** | **8,822** | **7,624** | **1,198 (13.6%)** | **6,481** | **1,143** |

Minority-class validation counts are now substantially larger than the
original split (e.g. Dark Spots: 61 → 151; Wrinkles: 100 → 154), which
directly addresses the reviewer's concern about unstable per-class metrics
on very small validation subsets.

## Known limitation to state explicitly

Perceptual-hash deduplication is a heuristic, not a certainty. A Hamming
distance ≤ 4 threshold was chosen and validated by manually inspecting
sampled pairs at each distance (all showed ≥0.90 normalized cross-
correlation), but a small number of genuinely distinct-but-visually-similar
images (e.g. sequential photos from the same clinical photo series) may have
been merged unnecessarily, and a small number of true near-duplicates below
that threshold may remain undetected. This is a reasonable and defensible
trade-off, but should be stated as a limitation rather than presented as a
perfect solution.
