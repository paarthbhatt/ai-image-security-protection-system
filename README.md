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
./.venv/bin/python -u src/run_pipeline.py \
    --real-dataset casia2 \
    --localise-limit 240 \
    --cross-dataset comofod \
    --cross-localise-limit 120 \
    --cross-variants 4
```

That single command runs the whole study end to end: load CASIA v2.0 → group
scenes → split → extract 143 features → train three models → select on
validation → evaluate on the held-out test set → sweep robustness → localise
regions against the ground-truth masks → score CoMoFoD zero-shot → write
figures, `results.json`, the model and the HTML dashboard.

Measured on this machine: **33 minutes cold** on 16 cores when
`FEATURE_VERSION` changed and every image must be re-extracted, and **11 m 49 s
warm** when the caches are reused (runs completed 2026-11-06). Dropping
`--localise-limit` removes every region metric and then also removes the
thresholds the transfer run inherits, so external localisation refuses to run
rather than tuning on data it should not touch.

Who does what, and the checks a number must pass before it enters the report, are
in [docs/TEAM-WORKFLOW.md](docs/TEAM-WORKFLOW.md).

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
| `figures/*.png` | The ten presentation figures, plus an editable SVG | `src/visualize.py` |
| `models/forensic_detector.joblib` | Selected model, threshold, feature names, provenance | `src/train.py` |
| `presentation/current_status.md` | Talk track and likely supervisor questions | `src/visualize.py` |
| `data/processed/features.npz` | Feature matrix cache, keyed by content digest | `src/train.py` |
| `data/processed/dataset_metadata.txt` | Corpus, seed, split rules, generator versions | `src/dataset.py` |

Figures are regenerated on every run: `dataset_distribution`, `roc_pr_curves`,
`confusion_matrix`, `model_comparison`, `feature_importances`, `robustness`,
`sample_pairs` (test images beside their ground-truth masks), `localisation`
(tiled against ELA, in-domain and per external corpus), `cross_dataset` (in-domain
beside zero-shot, per corpus) and `system_diagram` in both PNG and editable SVG.

## Feature groups

| Group | Count | What it measures | Literature basis |
|---|---:|---|---|
| `srm` | 72 | 8 high-pass residual kernels + inter-channel residuals; moments, entropy, sparsity, quantiles | Fridrich & Kodovský, IEEE TIFS 2012; Zhou et al., CVPR 2018 |
| `ela` | 16 | Error Level Analysis at q90/q75, block-wise dispersion, max/mean ratio | Krawetz, *A Picture's Worth* |
| `noise` | 5 | Wavelet MAD noise sigma at two scales + block dispersion | Donoho & Johnstone |
| `jpeg` | 4 | 8×8 blockiness, DCT high-frequency profile, AC zero ratio | JPEG-grid forensics |
| `copymove` | 8 | Duplicate 16×16 blocks matched by unit-normed low-frequency DCT descriptor, then clustered by displacement offset (block count, matched pairs, largest cluster, offset entropy, peak ratio) | Webswale & Shuller, SPIE 2007; classic CMFD |
| `global` + `chroma` | 38 | Grayscale/edge/chroma context and geometry | — |

Residuals matter because global colour moments mostly measure *content*, which a
brightness edit changes trivially.

## Measured results — CASIA v2.0

Real forgeries, not edits this project generated. 7,491 authentic / 5,123
tampered images; 6,523 linked scenes; splits 8,739 / 1,940 / 1,935.

| Arm | Features | ROC-AUC | F1 |
|---|---|---:|---:|
| As distributed (headline) | 143 | **0.943** (95% CI 0.933–0.953) | 0.845 |
| Forensic-only (`srm`+`noise`+`copymove`) | 85 | **0.698** | 0.611 |
| Compression-shaped only (`jpeg`+`ela`) | 20 | 0.887 | 0.777 |
| Content statistics only (`global`+`chroma`) | 38 | 0.909 | 0.798 |
| Format-normalised, retrained (both classes re-encoded JPEG q90; 400 per class in test) | 143 | 0.873 | 0.802 |

The headline row is the pipeline's own test block. The four group rows come from
`.work/diagnose_confound.py`, which fits the same model on the same frozen split
and picks its threshold from validation F1 candidates rather than the pipeline's
grid, so its all-feature AUC matches the headline and its F1 can differ in the
third decimal.

Per family on the held-out test set: splicing ROC-AUC 0.942 (n=650), copy-move
0.950 (n=101 — small, treat with care). Validation AUC per model: logistic
regression 0.871, random forest 0.940, histogram gradient boosting 0.947.

### Read the 0.943 with suspicion

CASIA stores authentic images as JPEG/BMP and 60% of the tampered images as
TIFF. A model can separate those containers without examining any forgery
evidence, and this one largely does: `jpeg`+`ela` alone reaches 0.887, and
content statistics alone reach 0.909. Remove every compression-flavoured group
and the forensically meaningful residual/noise/copy-move evidence supports only
**0.698**.

The honest reading: **~0.70 for genuine forensic signal, 0.943 when the
detector is allowed to use storage format as a shortcut.** Re-encoding every
image through one JPEG generation before training and testing retains 0.873
(`.work/normalize_probe.py`, 400 test images per class rather than the full
1,935, because it re-decodes every pixel). What that arm also shows is worth
stating: re-encoding both classes losslessly to PNG changes the score not at all
— identical to four decimals — because nothing in the feature vector can see a
file extension. The shortcut is not the container; it is what the container's
compression left in the pixels.

This is why the robustness sweep matters more than the headline:

| Condition | ROC-AUC | F1 |
|---|---:|---:|
| Clean | 0.952 | 0.862 |
| JPEG q90 | 0.842 | 0.695 |
| JPEG q70 | 0.816 | 0.665 |
| JPEG q50 | 0.739 | 0.615 |
| JPEG q30 | 0.729 | 0.607 |
| JPEG q20 | 0.730 | 0.609 |
| Half resolution | 0.807 | 0.686 |
| Blur | 0.827 | 0.689 |
| Noise | 0.920 | 0.809 |

q90 used to be the *worst* row — ROC-AUC 0.652 and F1 0.153 — which made no
sense for the mildest degradation. The cause was that the training augmentation
covered q70, q30, noise and blur, so q90 sat in a blind spot: the detector had
never seen a single-compression copy of anything. Adding `jpeg_q90` and
`jpeg_q50` to augmentation moved that row to 0.842 / 0.695 without touching a
single test image. Recorded here because it is the clearest example of a
robustness failure that was an artefact of the experiment, not of the model.

### The base paper's own protocol, run on our code

Yadav & Vishwakarma (2024, §4.1.3) do not evaluate on CASIA as distributed. They
re-encode every uncompressed image at a random JPEG quality factor and thin the
authentic class to 5,123 for balance. `src/modified_casia.py` reproduces that:
3,103 conversions (all 3,059 tampered TIFFs plus 44 of the 54 authentic BMPs),
quality factors 50–100 drawn per image from a hash so adding or removing one image
cannot reshuffle the others, 5,123/5,123 rows, and the build audit — including the
manifest's SHA-256 — in `reports/modified_casia_build.json`.

| Corpus | Rows | Selected model | ROC-AUC | F1 | MCC | Zero-shot CoMoFoD AUC |
|---|---:|---|---:|---:|---:|---:|
| CASIA as distributed | 12,614 | hist_gradient_boosting | 0.9430 | 0.8446 | 0.7399 | 0.5062 |
| Modified CASIA (paper protocol) | 10,246 | random_forest | 0.9211 | 0.8602 | 0.7260 | 0.5141 |

Two things follow, and the second is not what we expected. The parity corpus
scores 0.9211 where the distributed one scores 0.9430, and a different detector
wins. That gap is not a clean measurement of the format cue's value, and the
sentence should not be read as one: the parity corpus is also 2,368 rows smaller,
because balance there is achieved by *dropping* originals, and its test fold is a
different set of scenes. The controlled version of this number already exists —
0.698 with every compression-flavoured group removed from the same frozen split —
and the gap here is the ballpark price of working on a recompressed, balanced
corpus.

What the arm does settle is transfer. It does **not** improve it: 0.5141 is chance
by any reading, so tidying the training corpus is not the lever — the confound
lives on the evaluation side. The full causal test, including the per-geometry
breakdown on both corpora, is section 5 of
[docs/TRANSFER-FINDING.md](docs/TRANSFER-FINDING.md).

The arm runs under `--arm casia2_mod`, which relocates every artifact root to
`reports/arms/`, `figures/arms/` and `models/arms/` so a side experiment cannot
overwrite cited evidence. Note that the two rows are not directly comparable
numbers — different corpora, different row counts; they are the same protocol run
twice.

## Zero-shot transfer is at chance, and that is a result

The same frozen model, decision threshold and region thresholds were pointed at
CoMoFoD small v2 — 200 photographs it had never seen, of which at most 4 of each
scene's ~51 post-processing variants are scored:

| | In-domain CASIA | Zero-shot CoMoFoD |
|---|---:|---:|
| ROC-AUC | 0.943 | 0.506 |
| MCC | 0.740 | −0.003 |
| balanced accuracy | 0.877 | 0.499 |
| localisation Dice (tiled) | 0.131 | 0.051 |

Chance, not a bug. Training on CoMoFoD itself reaches 0.709 on held-out scenes,
the corpus audit passes against its own published composition, and all five of
its documented tamper geometries score 0.466–0.553 — so the failure is neither
labelling, nor an unlearnable corpus, nor a matcher shape. It is
post-processing: the never-reprocessed subset transfers at **0.825**, 19% of
the score's variance is explained by the post-processing code, and 0.0% by the
forgery label. Full mechanism, the eliminated alternatives, and what we refuse to
call transfer are in
[docs/TRANSFER-FINDING.md](docs/TRANSFER-FINDING.md).

## Evaluation protocol

Thresholds are chosen on validation, never on test. Reported: accuracy, balanced
accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, confusion matrix,
per-family ROC-AUC, percentile bootstrap CIs (400 replicates, grouped by scene),
and a group-leakage check that must report zero scenes crossing folds. Accuracy
alone is explicitly treated as insufficient.

Region localisation is scored separately from image-level screening, because they
answer different questions: 128-pixel tiles at stride 64 re-scored by the whole
image classifier, plus ELA tile energy, giving Dice, IoU, hit rate, pixel
accuracy and the falsely-flagged area on authentic images, at thresholds chosen
on validation only. External corpora inherit those thresholds rather than tuning
on data they are supposed to test.

## Limitations to state out loud

- **The headline number is confounded** by CASIA's file formats, as above.
  Quote 0.873 (format-normalised) or 0.698 (forensic-only), not 0.943, unless
  the caveat travels with the number.
- **Transfer to another corpus is at chance.** The zero-shot table above is the
  weakest result in this README and the most important one; a single-corpus
  accuracy number does not survive contact with a different camera, container or
  post-processing chain.
- **F1 collapses under re-compression** because the threshold is fixed from
  clean validation. ROC-AUC and PR-AUC are the threshold-free comparisons.
- **Localisation is tile rescoring, not segmentation.** Dice 0.131 with a hit
  rate of 0.883 means the tiles are usually suspicious and almost never correctly
  bounded; a whole-image classifier was never asked to draw a boundary. Real
  pixel masks need the U-Net that is not built yet.
- **No deep baseline.** A handcrafted-feature model cannot match a trained CNN
  on hard cases; none is included.
- **Copy-move support is thin** — 101 test images is not enough to claim much.
- **A single seed.** No multi-seed variance is reported yet.
- **Feature groups can die silently.** Until this week the eight copy-move
  features were identically zero in every run in this project's history, and
  still counted as part of "143 features". The silent `except Exception` that
  hid it is gone, and `.work/probe_inert_features.py` is in the rotation because
  a feature that never varies cannot be split on but does occupy a line in a
  table.
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
| Modified CASIA (5,123 / 5,123, random-QF) | Base-paper protocol, format-cue removal | Built from the CASIA archive by `src/modified_casia.py`, evaluated under `--arm casia2_mod` |
| CoMoFoD small (200 scenes) | Zero-shot external test | Loaded, audited against its published composition, scored — at chance, see docs/TRANSFER-FINDING.md |
| COVERAGE (100 scenes) | Untouched external test | Blocked: browser download with a shared password |
| Columbia (grayscale) | Untouched external test | Blocked: terms form, 403 without it |

Routes, sizes and gating for each are in
[docs/REAL-DATASETS.md](docs/REAL-DATASETS.md).

## References

- J. Fridrich and J. Kodovský, "Rich Models for Steganalysis of Digital Images," IEEE TIFS, 2012.
- P. Zhou et al., "Learning Rich Features for Image Manipulation Detection," CVPR 2018.
- B. Bayar and M. C. Stamm, "A Deep Learning Approach to Universal Image Manipulation Detection," ACM IH&MMSec 2016.
- J. Dong, W. Wang, T. Tan, "CASIA Image Tampering Detection Evaluation Database," IEEE ChinaSIP 2013.
- D. Tralic et al., "CoMoFoD — New Database for Copy-Move Forgery Detection," ELMAR 2013.
- B. Wen et al., "COVERAGE — A Novel Database for Copy-Move Forgery Detection," IEEE ICIP 2016.
- B. Webswale and M. Shuller, "Copy-Paste Detection Using Rank Transform," SPIE Digital Image Processing Applications 2007 — the block-matching lineage the `copymove` group now follows.
- R. S. Yadav and J. K. Vishwakarma, "A novel two-stream deep learning architecture for the detection of tampered image regions," Signal, Image and Video Processing 18(4):3341–3352, 2024. DOI 10.1007/s11760-024-02997-6 — the base paper this work replicates and extends.
- Dataset label/mask corrections: <https://github.com/SunnyHaze/IML-Dataset-Corrections>
