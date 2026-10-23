# AI-Based Image Security and Protection System

An evidence-first research prototype for **AI-assisted image integrity screening**.
The current milestone detects likely image manipulation, reports a calibrated risk
score, verifies byte-level integrity with SHA-256, and produces presentation-ready
figures and an audit record.

> **Current claim:** this is a research screening prototype, not proof of authenticity
> and not a replacement for a human forensic examiner.

## What exists in the current milestone

1. **Dataset acquisition:** attempts to download the CIFAR-10 image dataset from
   its official Toronto source. If the large archive is incomplete or the network
   is too slow for the presentation deadline, it automatically uses the local
   scikit-learn digits image set as a clearly labelled, credential-free pilot
   fallback. The actual source is recorded in `dataset_metadata.txt`.
2. **Controlled benchmark generation:** creates authentic images and balanced
   copy-move, splicing, brightness, and JPEG/post-processing manipulations. The
   source-image split is performed before transformations to avoid leakage.
3. **Interpretable baseline:** extracts image-forensic features (colour moments,
   gradients, local contrast, Laplacian noise, block inconsistency, and JPEG-like
   high-frequency energy) and trains a logistic-regression detector.
4. **Protection layer:** hashes the original file, encrypts/decrypts files with
   authenticated AES-GCM when the optional cryptography package is available, and
   writes a tamper-evident JSON audit record.
5. **Visual evidence:** generates class/tamper distributions, ROC and precision-
   recall curves, a confusion matrix, feature importance, sample image pairs, and
   an end-to-end system diagram.
6. **Presentation material:** `presentation/current_status.md` and a self-contained
   HTML dashboard in `reports/current_status.html`.

## Research scope

The defensible scope is **passive image-integrity screening** for two primary
manipulation families—copy-move and splicing—with post-processing robustness tests.
The project explicitly separates:

- **Integrity:** does the image appear manipulated?
- **Protection/provenance:** can we detect byte changes when a trusted reference
  hash exists, and can we protect a stored image with authenticated encryption?
- **Limitations:** the model cannot prove an image is genuine, recover every edit,
  or defeat an adaptive attacker.

### Research questions

1. Can lightweight forensic features distinguish authentic and manipulated images
   better than a majority-class baseline?
2. Which manipulation family is hardest to detect?
3. How much does JPEG/post-processing reduce performance?
4. Can the protection layer detect byte changes and preserve confidentiality?
5. Does performance remain useful when source images are held out rather than
   randomly mixed into train and test?

### Hypotheses

- H1: the forensic-feature model will substantially outperform the 50% majority
  baseline on the controlled benchmark.
- H2: copy-move and splicing will be easier to detect than subtle recompression.
- H3: performance will decrease after post-processing, which is why the evaluation
  reports robustness instead of accuracy alone.
- H4: SHA-256 verification will deterministically flag any byte-level change to a
  protected original, while AES-GCM will reject modified ciphertext.

## Datasets and how many are needed

### What is downloaded now

| Dataset | Role | Pilot size | Why it is used |
|---|---:|---:|---|
| CIFAR-10 train split | Preferred authentic image source | Up to 5,000 source images | Stable, public, passwordless download; supports a reproducible controlled benchmark |
| scikit-learn digits fallback | Fast presentation pilot source | Up to 1,797 source images | Local, credential-free fallback when the CIFAR-10 archive cannot finish downloading |
| Synthetic tamper benchmark | Manipulated counterpart of the held-out sources | 5,000 generated images by default | Balanced labels and known tamper type; enables exact evaluation and visual ground truth |

The prototype reports the synthetic nature and actual source of this benchmark. It
is **not** presented as a substitute for CASIA or CoMoFoD.

### Recommended final study

For a stronger research paper, use the following in addition to the pilot:

- **CASIA v2.0:** approximately 12,614 images, with authentic and tampered
  content. Use grouped splits and inspect known label/mask inconsistencies.
- **CoMoFoD small:** 200 copy-move scenes with post-processed variants. Do not
  count variants from one source scene as independent images.
- **CASIA v1.0 or COVERAGE:** keep one dataset untouched for external testing.

Recommended final minimum: **10,000–20,000 image instances, at least 300
independent source groups, balanced authentic/manipulated examples, plus an
external test set**. The pilot in this repository is designed to demonstrate the
pipeline and report honest baseline results today; it does not claim cross-dataset
generalisation.

### Dataset integrity rules

- Split by source image/group, never by individual transformed file.
- Keep an untouched test set until model selection is complete.
- Report both image count and independent source-group count.
- Deduplicate with file hashes/perceptual hashes in the final study.
- Audit resolution, compression, metadata, and class balance before training.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Downloads CIFAR-10, builds the balanced benchmark, trains and evaluates.
python src/run_pipeline.py --source-images 500

# Reuse downloaded data and run a faster smoke-sized experiment.
python src/run_pipeline.py --source-images 500
```

Outputs are written to:

- `data/raw/` — downloaded archive and extracted source images
- `data/processed/benchmark.csv` — image-level labels and source-group IDs
- `data/processed/features.npz` — numeric feature matrix and labels
- `models/forensic_detector.joblib` — trained model and metadata
- `figures/` — PNG/SVG figures used in the presentation
- `reports/results.json` — machine-readable metrics and protection checks
- `reports/current_status.html` — visual status dashboard

## Evaluation

The pipeline reports:

- Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- Confusion matrix and ROC/PR curves
- Per-tamper-type recall and error counts
- Top feature coefficients as an interpretable baseline explanation
- SHA-256 hash verification and AES-GCM tamper rejection checks

Accuracy is not treated as sufficient evidence. A complete study should also add
calibration, confidence intervals, multiple random seeds, external datasets,
inference time, and robustness by JPEG quality/resizing/cropping severity.

## Citations and external datasets for the next milestone

- J. Dong, W. Wang, and T. Tan, “CASIA Image Tampering Detection Evaluation
  Database,” IEEE ChinaSIP, 2013. Public mirror/ground truth:
  <https://github.com/namtpham/casia2groundtruth>
- D. Tralic et al., “CoMoFoD—New Database for Copy-Move Forgery Detection,”
  ELMAR, 2013. <https://www.vcl.fer.hr/comofod/>
- B. Wen et al., “COVERAGE—A Novel Database for Copy-Move Forgery Detection,”
  IEEE ICIP, 2016. <https://github.com/wenbihan/coverage>
- B. Bayar and M. C. Stamm, “A Deep Learning Approach to Universal Image
  Manipulation Detection Using a New Convolutional Layer,” ACM IH&MMSec, 2016.
  <https://doi.org/10.1145/2909827.2930786>
- CIFAR-10 official page: <https://www.cs.toronto.edu/~kriz/cifar.html>

Known dataset quality corrections for several image-manipulation benchmarks:
<https://github.com/SunnyHaze/IML-Dataset-Corrections>

## Limitations to state during the presentation

This controlled pilot uses synthetic manipulations and small 32×32 images. A
model can learn dataset shortcuts such as compression or colour differences. The
results demonstrate a functioning research pipeline, not production readiness.
Real-world validation should use CASIA/CoMoFoD plus an unseen external dataset,
grouped splits, stronger deep baselines, and pixel-level masks for localization.
