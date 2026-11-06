# AI-Based Image Security and Protection System

An evidence-first research prototype for **image-manipulation screening**.
It detects likely tampering from passive forensic evidence, reports a risk score,
verifies byte-level integrity with SHA-256, optionally protects stored images with
authenticated AES-GCM, and produces audit-ready figures and a dashboard.

> **Current claim:** a research screening prototype evaluated on a real forgery
> benchmark. Not proof of authenticity, and not a replacement for a human
> forensic examiner.

## Quick start

```bash
cd researchproject
./.venv/bin/python -u src/run_pipeline.py --real-dataset casia2
```

That single command runs the whole study end to end: load CASIA v2.0 → group
scenes → split → extract 143 features → train three models → select on
validation → evaluate on the held-out test set → sweep robustness → write
figures, `results.json`, the model and the HTML dashboard.

Roughly **2 hours cold** on 16 cores (feature extraction dominates) and about
**20 minutes warm**, when the feature caches are reused.

Do not pipe the run through `tail`/`head` while watching it: they buffer and the
run looks frozen. Either drop the pipe or use `python -u`.

Fresh machine setup:

```bash
python3 -m venv .venv
. .venv/bin/activate.fish        # fish; bash uses: source .venv/bin/activate
pip install -r requirements.txt
```

CASIA v2.0 must already be in `data/real/casia2/` with the `Au/`, `Tp/`, `Gt/`
layout — see [docs/REAL-DATASETS.md](docs/REAL-DATASETS.md). Its size and licence
mean it is never committed.

## What actually runs

1. **Load the corpus.** `src/sources.py` reads `Au/` (authentic), `Tp/`
   (tampered) and `Gt/` (ground-truth masks), and maps the letter after `Tp_` to
   a forgery family: `D` → copy-move, `S` → splicing.
2. **Group scenes, then split.** CASIA names each forgery after the photographs
   it was built from, so `Au_ani_00018.jpg` and every `Tp_*_ani00018_*` are the
   same scene. Those are linked transitively into one `source_id`, and folds are
   assigned by whole group. 12,614 images collapse to **6,523 scenes**.
3. **Extract features.** 143 values per image across six literature-grounded
   groups (below), computed in parallel across cores.
4. **Train and select.** Logistic regression, random forest and histogram
   gradient boosting are fitted on the training split; the winner is chosen on
   **validation ROC-AUC only**. Training rows are expanded with degraded copies
   of themselves so the detector is not tuned to one compression state.
5. **Evaluate.** Held-out test metrics, grouped percentile-bootstrap confidence
   intervals, per-family breakdown, a group-leakage check, and a robustness
   sweep over nine conditions.
6. **Protect and report.** SHA-256 / AES-GCM checks, figures, `results.json`,
   and a self-contained HTML dashboard.

## Where the results appear

| Path | What it holds | Written by |
|---|---|---|
| `reports/current_status.html` | Self-contained dashboard — **open this first** | `src/visualize.py` |
| `reports/results.json` | Every metric, CI, leakage check, robustness row | `src/train.py` |
| `figures/*.png` | The nine presentation figures | `src/visualize.py` |
| `models/forensic_detector.joblib` | Selected model, threshold, feature names, provenance | `src/train.py` |
| `presentation/current_status.md` | Talk track and likely supervisor questions | `src/visualize.py` |
| `data/processed/features.npz` | Feature matrix cache, keyed by content digest | `src/train.py` |
| `data/processed/dataset_metadata.txt` | Corpus, seed, split rules, generator versions | `src/dataset.py` |

Figures are regenerated on every run: `dataset_distribution`, `roc_pr_curves`,
`confusion_matrix`, `model_comparison`, `feature_importances`, `robustness`,
`sample_pairs` (test images beside their ground-truth masks), and
`system_diagram` in both PNG and editable SVG.

## Feature groups

| Group | Count | What it measures | Literature basis |
|---|---:|---|---|
| `srm` | 72 | 8 high-pass residual kernels + inter-channel residuals; moments, entropy, sparsity, quantiles | Fridrich & Kodovský, IEEE TIFS 2012; Zhou et al., CVPR 2018 |
| `ela` | 16 | Error Level Analysis at q90/q75, block-wise dispersion, max/mean ratio | Krawetz, *A Picture's Worth* |
| `noise` | 5 | Wavelet MAD noise sigma at two scales + block dispersion | Donoho & Johnstone |
| `jpeg` | 4 | 8×8 blockiness, DCT high-frequency profile, AC zero ratio | JPEG-grid forensics |
| `copymove` | 8 | ORB self-matching keypoints, offset clustering, peak-cluster ratio | Classic CMFD |
| `global` + `chroma` | 38 | Grayscale/edge/chroma context and geometry | — |

Residuals matter because global colour moments mostly measure *content*, which a
brightness edit changes trivially.

## Measured results — CASIA v2.0

Real forgeries, not edits this project generated. 7,491 authentic / 5,123
tampered images; 6,523 linked scenes; splits 8,739 / 1,940 / 1,935.

| Arm | Features | ROC-AUC | F1 |
|---|---|---:|---:|
| As distributed (headline) | 143 | **0.947** (95% CI 0.938–0.956) | 0.850 |
| Forensic-only (`srm`+`noise`+`copymove`) | 85 | **0.671** | 0.573 |
| Compression-shaped only (`jpeg`+`ela`) | 20 | 0.903 | 0.793 |
| Content statistics only (`global`+`chroma`) | 38 | 0.926 | 0.819 |
| Format-normalised, retrained (all images re-encoded JPEG q90) | 143 | 0.868 | 0.809 |

Per family on the held-out test set: splicing ROC-AUC 0.947 (n=650), copy-move
0.951 (n=101 — small, treat with care). Validation AUC per model: logistic
regression 0.871, random forest 0.943, histogram gradient boosting 0.947.

### Read the 0.947 with suspicion

CASIA stores authentic images as JPEG/BMP and 60% of the tampered images as
TIFF. A model can separate those containers without examining any forgery
evidence, and this one largely does: `jpeg`+`ela` alone reaches 0.903, and
content statistics alone reach 0.926. Remove every compression-flavoured group
and the forensically meaningful residual/noise/copy-move evidence supports only
**0.671**.

The honest reading: **~0.67 for genuine forensic signal, 0.947 when the
detector is allowed to use storage format as a shortcut.** Re-encoding everything
to a common JPEG quality before training and testing retains 0.868, which is
what the deployed pipeline would actually see on re-compressed images.

This is why the robustness sweep matters more than the headline:

| Condition | ROC-AUC | F1 |
|---|---:|---:|
| Clean | 0.958 | 0.875 |
| JPEG q90 | 0.840 | 0.693 |
| JPEG q70 | 0.812 | 0.638 |
| JPEG q50 | 0.723 | 0.584 |
| JPEG q30 | 0.721 | 0.584 |
| JPEG q20 | 0.710 | 0.611 |
| Half resolution | 0.813 | 0.696 |
| Blur | 0.826 | 0.682 |
| Noise | 0.921 | 0.804 |

q90 used to be the *worst* row — ROC-AUC 0.652 and F1 0.153 — which made no
sense for the mildest degradation. The cause was that the training augmentation
covered q70, q30, noise and blur, so q90 sat in a blind spot: the detector had
never seen a single-compression copy of anything. Adding `jpeg_q90` and
`jpeg_q50` to augmentation moved that row to 0.840 / 0.693 without touching a
single test image. Recorded here because it is the clearest example of a
robustness failure that was an artefact of the experiment, not of the model.

## Evaluation protocol

Thresholds are chosen on validation, never on test. Reported: accuracy, balanced
accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, per-family
ROC-AUC, percentile bootstrap CIs (400 replicates, grouped by scene), and a
group-leakage check that must report zero scenes crossing folds. Accuracy alone
is explicitly treated as insufficient.

## Limitations to state out loud

- **The headline number is confounded** by CASIA's file formats, as above.
  Quote 0.868 (format-normalised) or 0.671 (forensic-only), not 0.957, unless
  the caveat travels with the number.
- **F1 collapses under re-compression** because the threshold is fixed from
  clean validation. ROC-AUC and PR-AUC are the threshold-free comparisons.
- **No localisation metric yet.** Ground-truth masks are loaded and rendered,
  but IoU/Dice are not scored.
- **No deep baseline.** A handcrafted-feature model cannot match a trained CNN
  on hard cases; none is included.
- **Copy-move support is thin** — 101 test images is not enough to claim much.
- **A single seed.** No multi-seed variance is reported yet.
- SHA-256 proves byte equality against a trusted reference created *before*
  distribution. AES-GCM protects stored content. Neither makes an image genuine.

## Synthetic mode (development only)

A generated benchmark still exists for pipeline work: CIFAR-10 sources with
alpha-feathered copy-move, splicing and removal edits, one manipulated
counterpart per source, splits grouped by source id *before* manipulation.

```bash
python src/run_pipeline.py --dataset cifar10 --source-images 5000
python src/run_pipeline.py --real-dataset casia2 --max-images 1500   # smoke test
```

`--max-images` caps a run to answer "does this work", never "how good is it".
Conclusions in this README come only from the full CASIA run.

## Other real benchmarks

| Dataset | Role | Status |
|---|---|---|
| CASIA v2.0 (7,491 Au / 5,123 Tp) | Primary evidence | Loaded and evaluated |
| CoMoFoD small (200 scenes) | Copy-move + post-processing | Manual download, see docs |
| COVERAGE (100 scenes) | Untouched external test | Manual download, see docs |

Routes, sizes and gating for each are in
[docs/REAL-DATASETS.md](docs/REAL-DATASETS.md).

## References

- J. Fridrich and J. Kodovský, "Rich Models for Steganalysis of Digital Images," IEEE TIFS, 2012.
- P. Zhou et al., "Learning Rich Features for Image Manipulation Detection," CVPR 2018.
- B. Bayar and M. C. Stamm, "A Deep Learning Approach to Universal Image Manipulation Detection," ACM IH&MMSec 2016.
- J. Dong, W. Wang, T. Tan, "CASIA Image Tampering Detection Evaluation Database," IEEE ChinaSIP 2013.
- D. Tralic et al., "CoMoFoD — New Database for Copy-Move Forgery Detection," ELMAR 2013.
- B. Wen et al., "COVERAGE — A Novel Database for Copy-Move Forgery Detection," IEEE ICIP 2016.
- Dataset label/mask corrections: <https://github.com/SunnyHaze/IML-Dataset-Corrections>
