# Retraining M1-M5 on the clean, deduplicated dataset

This folder is already inside your `SkinCAW-Repo` folder (at `retrain/`), next
to `dataset/SKIN_CLEAN_DEDUP/` (the rebuilt, leakage-free dataset) and
`dataset/SKIN_PROJECT_CLEANED2_edgemix/` (the original, leaky one, kept for
reference).

## 1. One-time setup (Terminal, on your Mac)

```bash
cd ~/Documents/SkinCAW-Repo/retrain
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch with Apple Silicon (MPS) support automatically on an M-series Mac — no special flags needed.

Verify MPS is available:
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```
Should print `True`. If it prints `False`, training will still work but fall back to CPU (much slower) — let me know and we'll troubleshoot.

## 2. Smoke test (run this before anything else — takes under a minute)

```bash
python smoke_test.py --data_root ../dataset/SKIN_CLEAN_DEDUP
```

This runs one tiny batch through all 8 configurations (M1-M5 plus the three ablation-isolation variants) and prints PASS/FAIL for each. If anything fails, send me the exact error message before running full training — much cheaper to fix now than an hour into a real run.

## 3. Time a real epoch before committing

```bash
python train.py --stage m1 --data_root ../dataset/SKIN_CLEAN_DEDUP --out_dir ../results/m1_test --epochs 1
```

Time this one epoch. Multiply by how many epochs you plan (20 is the default) to estimate total time for that stage. On an M3 MacBook with MPS, a ResNet-50 epoch over ~6,500 training images at batch size 32 typically takes somewhere in the range of 1-5 minutes, but this varies — **time it yourself rather than trusting a generic estimate.** If it's too slow given the Sept 10 deadline, lower `--epochs` and `--patience`, or drop to batch size 16.

## 4. Run the real training

**Interruptions are safe.** Every stage saves a full `checkpoint.pt` (model + optimizer state) after *every* epoch, not just when it improves. If you Ctrl+C, close the terminal, or the Mac sleeps mid-stage, just re-run the exact same `train.py` command (or re-run `run_all.sh`) and it picks up from the last completed epoch instead of starting over. A stage that already finished will detect that and skip straight to final evaluation, so it's also safe to re-run `run_all.sh` after some stages already completed. Recommend running with `caffeinate -i` so macOS doesn't sleep mid-run:

```bash
caffeinate -i bash run_all.sh
```

This runs M1 → M2 → M3 → M4 → M5 in sequence (the paper's main narrative, now on one consistent pipeline). Each stage's results land in `../results/<stage>/`:
- `metrics.json` — accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix
- `val_predictions.csv` — every validation image's true label, predicted label, and per-class probabilities (needed for the statistics below)
- `history.json` — per-epoch loss/accuracy curves (for the loss-curve figure)
- `best_model.pt`, `final_model.pt` — checkpoints
- `run_config.json` — exact hyperparameters used (for the paper's reproducibility)

**If you have spare time before the deadline**, uncomment the three `m5_caw_only` / `m5_contrast_only` / `m5_gabor_only` lines at the bottom of `run_all.sh` and run those too — they isolate each component of M5 individually, which is exactly the controlled ablation both reviewers asked for (comment #4 in the review doc, comment #7 in the PDF). Not required, but it meaningfully strengthens the resubmission.

Feel free to run stages in parallel in separate Terminal tabs if you want to save wall-clock time — just don't run more than 2 at once, or memory pressure will slow both down.

## 5. Statistics (after at least two stages have finished)

McNemar's test (reviewer comment #13 in the PDF), comparing e.g. M1 vs M5:
```bash
python stats.py mcnemar ../results/m1/val_predictions.csv ../results/m5/val_predictions.csv
```

Exact binomial confidence interval for any n/N result (e.g. the external test set):
```bash
python stats.py binomial_ci <n_correct> <n_total>
```

## 6. External test set (the 12/27 "unseen" images)

```bash
python external_eval.py --checkpoint ../results/m5/best_model.pt --use_caw \
    --images_dir ../dataset/randomm/images
```

This infers ground truth from filenames (acne/eczema/rosacea/wrinkle/dark spot/normal) and reports accuracy automatically — **please sanity-check the printed true/predicted pairs against the actual images**, since filename-based labeling is a heuristic, not an authoritative ground-truth file (worth being upfront about in the manuscript too).

## 7. IoU / Grad-CAM++ evaluation (the interpretability numbers)

```bash
python iou_eval.py --checkpoint ../results/m5/best_model.pt --use_caw \
    --randomm_dir ../dataset/randomm
```

Reports per-image and mean IoU, plus a center-crop baseline and a random-mask baseline for comparison — the reviewer specifically asked for these baselines (review doc, comment #9).

## 8. Qualitative figures (Grad-CAM comparison, IoU breakdown, loss curves, Gabor scatter, CAW concept maps)

```bash
python make_qualitative_figures.py --results_dir ../results --randomm_dir ../dataset/randomm \
    --data_root ../dataset/SKIN_CLEAN_DEDUP --out_dir ../figures_regenerated
```

Requires all of `results/m1` through `results/m5` (best_model.pt) and `results/m5/history.json` to exist. Produces 5 PNGs in `../figures_regenerated/`:
- `fig5_gradcam_comparison.png` — Original / M1–M5 Grad-CAM panel for one representative image (matches the layout of the qualitative comparison figure from the earlier version of this paper, but regenerated from the retrained checkpoints)
- `fig6_iou_breakdown.png` — Image / GT mask / thresholded CAM for the best, median, and worst IoU images in `dataset/randomm` (GT masks are human annotations and are unchanged; only the thresholded-CAM column is regenerated from M5)
- `fig7a_loss_curves.png` — Train/val CE, Gabor, and contrastive loss curves for the unified M5 model
- `fig7b_gabor_scatter.png` — Predicted vs. true Gabor energy on the M5 validation set, with Pearson r printed to the terminal and in the figure title
- `fig8_caw_concepts.png` — Original / Grad-CAM++ / CAW Concept-0 spatial activation map for 2 example images, computed from the corrected CAW module (real per-channel concept attention, not the collapsed-scalar bug in the earlier version)

## 9. What to send back to me

Once stages finish, just tell me (or paste/share):
- the `metrics.json` for each stage you ran
- the McNemar's and binomial CI outputs
- the external eval and IoU eval JSON outputs

I'll fold the real numbers into the revised manuscript, the results tables, and the response-to-reviewers letter. You don't need to send the checkpoint files (large) unless you want me to also regenerate the Grad-CAM figures — the JSON/CSV outputs are enough for the statistics and tables.

## What changed vs. the original notebooks (for your own context)

- **One framework, one pipeline** for all 5 stages (previously M1-M4 were TensorFlow/Keras, M5 was PyTorch — different frameworks make the comparison uncontrolled, which was a major reviewer complaint).
- **Leakage fixed**: 14.3% of the old validation set was a duplicate or near-duplicate of a training image. The new split (`SKIN_CLEAN_DEDUP`) has zero overlap, verified programmatically.
- **Consistent Gabor targets**: the old repo had three different, mutually inconsistent Gabor-energy formulas across notebooks (and train/val didn't even match each other). Recomputed once, consistently, for the new split.
- **Fixed a bug in the CAW attention module**: the original `torch.einsum` computation silently collapsed the 4 concept-similarity scores into one scalar gate applied uniformly across all channels, discarding the entire point of having multiple concepts. Fixed to produce genuine per-channel attention.
- **One coherent M5**: the original repo had two different final checkpoints — one with the attention module but no Gabor loss, one with the Gabor loss but no attention module — and the paper's Results section describes them as if they were the same model. `common.py`'s `m5` preset trains the real thing: attention + contrastive + Gabor, together, on one model.
- **Image normalization restored**: the original M5 training loop didn't apply ImageNet mean/std normalization at all, despite the Methods section saying it did. Now consistent across every stage.
