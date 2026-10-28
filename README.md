# AI-Based Image Security and Protection System

An evidence-first research prototype for **image-manipulation screening**.
Detects likely tampering from passive forensic evidence, reports a calibrated
risk score, verifies byte-level integrity with SHA-256, optionally protects
stored images with authenticated AES-GCM, and produces audit-ready figures.

> **Current claim:** a research screening prototype, not proof of authenticity and
> not a replacement for a human forensic examiner.

## What actually runs

1. **Authentic-image corpus.** CIFAR-10 (natural images, 32×32) by default. The
   archive is downloaded to `.part`, verified for completeness, and only then
   promoted — an interrupted transfer raises instead of silently degrading.
   `scikit-learn` digits exists as an emergency pilot corpus and requires an
   explicit `--allow-fallback`.
2. **Controlled manipulation benchmark.** Alpha-feathered copy-move, splicing and
   object-removal edits with **ground-truth masks**, one manipulated counterpart
   per authentic source image. Splits are grouped by source id *before*
   manipulation, so no scene leaks across folds.
3. **Real forensic benchmarks.** CASIA v2.0, CoMoFoD and COVERAGE load from
   `data/real/<name>/` with layout auto-detection and scene-level grouping. See
   [docs/REAL-DATASETS.md](docs/REAL-DATASETS.md). These are the only valid
   evidence of real-world performance.
4. **Forensic detector.** 143 features in five literature-grounded groups
   (below), compared across logistic regression, random forest and histogram
   gradient boosting; the winner is selected on validation ROC-AUC only.
5. **Protection layer.** SHA-256 hashing, optional AES-GCM encryption, and a
   tamper-evident JSON audit record.
6. **Evidence.** ROC/PR curves, confusion matrix, model comparison, per-family
   breakdown, feature importances, robustness sweep, sample pairs, and a
   self-contained HTML dashboard.

## Feature groups

| Group | What it measures | Literature basis |
|---|---|---|
| `srm` | 8 high-pass residual kernels + 3 inter-channel residuals; moments, entropy, sparsity, quantiles per channel | Fridrich & Kodovský, *Rich Models*, IEEE TIFS 2012; Zhou et al., CVPR 2018 |
| `ela` | Error Level Analysis at q90/q75, incl. block-wise dispersion and max/mean ratio | Krawetz, *A Picture's Worth* (ELA) |
| `noise` | Wavelet MAD noise sigma at two scales + spatial dispersion of block estimates | Donoho & Johnstone; noise-inconsistency splicing detectors |
| `jpeg` | 8×8 blockiness ratio, DCT high-frequency profile, AC zero ratio | JPEG-grid forensics |
| `copymove` | ORB self-matching keypoints, offset clustering, peak-cluster ratio | Classic CMFD (SIFT/ORB + clustering) |
| `global` | Grayscale/edge/chroma context and geometry | — |

The SRM group is the important addition: global colour moments mostly measure
*content*, which a brightness edit changes trivially. Residuals suppress content
and expose the noise and quantisation traces an edit disturbs.

## What changed from the earlier pilot

- **Provenance is derived, not assumed.** `benchmark.csv` previously wrote
  `cifar10_*` source ids for images that came from the digits fallback. Source
  ids and a `dataset` column now come from the loader that produced the pixels.
- **Fallback is fatal.** A truncated archive used to print a warning and continue.
  It now raises `DatasetUnavailable` unless `--allow-fallback` is passed.
- **Brightness and JPEG are no longer forgery classes.** They are not
  manipulations; they are post-processing. They moved into the robustness sweep,
  where they belong.
- **Masks are generated and stored** for every manipulated image, enabling
  localisation metrics.
- **One model became three**, with validation-only selection, bootstrap
  confidence intervals, and an explicit group-leakage check.
- **Training is hardened against recompression.** Without it the detector
  collapsed to ROC-AUC 0.50 the moment an image was re-saved as JPEG. Training
  rows are now expanded with JPEG-q70/q30, noise and blur variants; feature
  matrices are cached so a repeat run costs seconds, not minutes.
- **Extraction is now fast enough for full-resolution benchmarks.** Copy-move
  keypoints are searched at a fixed 256 px working scale (5.6x faster, same
  evidence: a copy-move pair shares a translation, not a scale) and extraction
  runs across all cores. A 5,000-source run went from 760 s to 136 s with
  identical metrics, and a 768x512 photograph costs ~0.55 s instead of 2.4 s.
  That is the difference between a real-data run taking days and taking hours.

## Measured results (CIFAR-10, 10,000 instances, 5,000 source groups)

| Condition | ROC-AUC |
|---|---:|
| Clean held-out test | **0.657** (95% CI 0.632–0.681) |
| Splicing only | 0.708 |
| Copy-move only | 0.635 |
| Removal only | 0.626 |
| JPEG q70 | 0.605 |
| JPEG q20 | 0.515 |
| Half resolution | 0.512 |

Read this honestly: a 143-feature handcrafted detector reaches only modest
separation on 32×32 natural images. Earlier reported scores near 0.94 came from
the digits fallback, where brightness and JPEG were themselves the labels — a
shortcut, not forgery detection. Feature importances confirm the shift: the top
signals are now SRM residuals and gradient statistics, not global colour means.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic benchmark on real natural images (~2 min cold, ~50 s cached)
python src/run_pipeline.py --dataset cifar10 --source-images 5000

# Do NOT pipe through `tail`/`head` while watching progress: they buffer and the
# run looks frozen. Drop the pipe, or use `python -u`:
python -u src/run_pipeline.py --dataset cifar10 --source-images 5000

# Three seeds, aggregated
python src/run_pipeline.py --dataset cifar10 --source-images 5000 --seeds 42,7,13

# A real forgery benchmark (see docs/REAL-DATASETS.md)
python src/run_pipeline.py --real-dataset casia2
```

Outputs:

- `data/raw/` — verified archive and extracted source images
- `data/processed/benchmark.csv` — labels, source groups, splits, mask paths
- `data/processed/masks/` — ground-truth manipulation masks
- `data/processed/features.npz` — feature matrix, labels, splits
- `models/forensic_detector.joblib` — selected model, threshold, provenance
- `figures/` — presentation figures
- `reports/results.json` — metrics, CIs, leakage check, robustness
- `reports/current_status.html` — self-contained dashboard

## Evaluation protocol

Thresholds are chosen on validation, never test. Reported: accuracy, balanced
accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, per-family
ROC-AUC, percentile bootstrap CIs (400 replicates), and a group-leakage check.
Accuracy alone is explicitly treated as insufficient.

## Datasets

| Dataset | Role | Status |
|---|---|---|
| CIFAR-10 train split | Preferred authentic source | Automatic, verified download |
| Synthetic tamper benchmark | Controlled counterpart | Generated with masks |
| CASIA v2.0 (7,491 Au / 5,123 Tp) | Development | Manual, see docs |
| CoMoFoD small (200 scenes) | Copy-move + post-processing | Manual, see docs |
| COVERAGE (100 scenes) | Untouched external test | Manual, see docs |
| scikit-learn digits | Emergency pilot only | Explicit opt-in |

Final-study target: 10,000–20,000 instances from ≥300 independent source groups,
plus an external test set that is never used for model selection.

## Limitations to state out loud

- The synthetic benchmark is built from *your* manipulations, not a human
  forger's. It bounds what you can claim.
- CIFAR-10 is 32×32. Many forensic traces (JPEG grids, sensor noise) barely
  exist at that size. Real datasets are full resolution and behave differently.
- A handcrafted-feature model cannot compete with a trained CNN on tough cases.
  No deep baseline is included yet.
- No localisation metric is reported, even though masks are stored.
- SHA-256 proves byte equality only against a trusted reference created *before*
  distribution. AES-GCM protects stored content; neither makes an image genuine.

## References

- J. Fridrich and J. Kodovský, "Rich Models for Steganalysis of Digital Images,"
  IEEE TIFS, 2012.
- P. Zhou et al., "Learning Rich Features for Image Manipulation Detection,"
  CVPR 2018.
- B. Bayar and M. C. Stamm, "A Deep Learning Approach to Universal Image
  Manipulation Detection Using a New Convolutional Layer," ACM IH&MMSec 2016.
- J. Dong, W. Wang, T. Tan, "CASIA Image Tampering Detection Evaluation
  Database," IEEE ChinaSIP 2013.
- D. Tralic et al., "CoMoFoD — New Database for Copy-Move Forgery Detection,"
  ELMAR 2013.
- B. Wen et al., "COVERAGE — A Novel Database for Copy-Move Forgery Detection,"
  IEEE ICIP 2016.
- Dataset label/mask corrections:
  <https://github.com/SunnyHaze/IML-Dataset-Corrections>
- CIFAR-10: <https://www.cs.toronto.edu/~kriz/cifar.html>
