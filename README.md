# Supplementary material — Multi-Class Skin Disease Classification with Gabor-Supervised Concept-Guided Attention

This folder contains the code, results, and manifests needed to reproduce the
numbers reported in the paper. It is referenced from the paper's
Declarations section ("Availability of Data and Material") and the
Limitations section ("supplementary audit report").

## What's here

- **`code/`** — the training and evaluation pipeline (`train.py`, `common.py`,
  `stats.py`, `iou_eval.py`, `external_eval.py`, `smoke_test.py`,
  `run_all.sh`, `requirements.txt`). Running `run_all.sh` against the dataset
  manifest below reproduces M1–M5 and the three ablation-isolation
  configurations (`m5_caw_only` / `m5_contrast_only` / `m5_gabor_only` — note:
  `caw` is CGA's pre-rename internal flag name, documented in the paper,
  Section III-B1).
- **`results/<stage>/`** — for each of the 8 trained configurations
  (M1–M5 plus the three isolated-component ablations): `metrics.json`
  (accuracy, macro/weighted F1, per-class precision/recall/F1, confusion
  matrix), `history.json` (per-epoch loss/accuracy curves), `val_predictions.csv`
  (every validation image's true label, predicted label, and per-class
  probabilities — this is what `stats.py` consumes to compute McNemar's test),
  and `run_config.json` (exact hyperparameters used for that run).
- **`dataset_manifest/`** — `manifest.csv` (the exact 7,624-image
  train/validation split used in this study, i.e. which file went to which
  split), `dataset_audit_report.md` (the deduplication audit: near-duplicate
  clusters found, method, validation performed), and `gabor_targets.json`
  (the precomputed Gabor-energy regression targets, training-set-normalized).
- **`stats/`** — `iou_results.json` (per-image Grad-CAM++/center-crop/random
  IoU values for all 26 externally-evaluated images — this is what the
  paired t-test and Wilcoxon signed-rank test in Section IV-E were computed
  from) and `external_eval_results.json` (the 26-image external evaluation's
  per-image predictions).
- **`figures/`** — the final, submission-quality figures (fig1–fig9,
  corresponding to the paper's Figures 1, 2, 3, 4(a), 4(b), 5, 6, 7(a), 7(b)
  in that order), 600+ DPI.

## What's deliberately NOT included

- **Model checkpoints (`.pt` files).** Each stage's `best_model.pt` /
  `final_model.pt` is ~96 MB and `checkpoint.pt` (includes optimizer state)
  is ~285 MB; across 8 stages that's several GB, over GitHub's 100 MB
  per-file limit. If you want to publish these too, use Git LFS or a
  separate archive (Zenodo, HuggingFace Hub) and link it here — happy to
  help set that up if you want it.
- **The training/validation images themselves.** As stated in the paper
  (Section III-A, Declarations), the dataset is assembled from several
  public sources under non-commercial research licenses and is not
  redistributed; `manifest.csv` records which file each source contributed
  without reproducing the images.
- **The 26-image external evaluation set**, for the same licensing/
  redistribution reasons stated in the paper (Section IV-G).

## Reproducing the paper's numbers

```bash
cd code
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # trains M1-M5; uncomment the three ablation lines for Table 5
python stats.py            # McNemar's test, confidence intervals (Section IV-A, IV-G)
python iou_eval.py         # Grad-CAM++ IoU + significance tests (Section IV-E)
```

`run_all.sh` expects a dataset laid out the same way as
`dataset_manifest/manifest.csv` describes (`SKIN_CLEAN_DEDUP/train/<class>/`
and `SKIN_CLEAN_DEDUP/val/<class>/`); since the images aren't redistributed
here, you'll need to assemble them from the sources cited in the paper
(Section III-A) and arrange them according to the manifest.
