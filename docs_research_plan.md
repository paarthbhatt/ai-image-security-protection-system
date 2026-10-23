# Research plan: AI-Based Image Security and Protection System

## 1. Study design

This is an **applied, experimental, quantitative design-science study**. The
software artifact is a reproducible image-integrity screening system; the research
contribution is the controlled comparison of a lightweight forensic baseline across
manipulation families and post-processing conditions.

### System boundary

**In scope:** authentic/manipulated classification, copy-move and splicing
simulation, JPEG/post-processing robustness, source-group-safe evaluation,
SHA-256 integrity verification, authenticated encryption, audit logging, and
presentation-ready evidence.

**Out of scope for this milestone:** universal deepfake detection, source-camera
attribution, legal proof, adaptive adversaries, full pixel-level segmentation, and
production cloud deployment.

## 2. Threat model

An attacker can alter image pixels, copy a region, insert content from another
image, re-save at a lower JPEG quality, resize, blur, or add noise. The defender
may have only the submitted image and may not have an original reference.

The model returns a probability-like risk score and should route suspicious images
to human review. A cryptographic hash provides proof of equality only when a
trusted reference hash was created before distribution. AES-GCM protects stored
content and detects modified ciphertext; it does not make a manipulated image
genuine.

## 3. Research questions and hypotheses

| Question | Evidence in this milestone |
|---|---|
| Can forensic features separate authentic and manipulated images? | ROC-AUC, PR-AUC, balanced accuracy, F1 and confusion matrix |
| Which manipulation type is hardest? | Per-type recall and error table |
| What is the effect of post-processing? | JPEG/resize/blur variants and robustness figure |
| Does the protection layer detect changes? | Hash mismatch and AES-GCM tamper rejection checks |

H1 predicts performance above the majority baseline. H2 predicts easier detection
of copy-move/splicing than subtle recompression. H3 predicts a robustness drop.

## 4. Data protocol

The pilot selects source images from CIFAR-10, splits the sources before creating
manipulations, and creates one authentic plus one manipulated record per source.
The test set therefore contains source images not used to make training examples.

For the final study, use CASIA v2.0 and CoMoFoD small for development, and hold
CASIA v1.0 or COVERAGE out for external testing. Count independent scenes/groups,
not merely post-processed files. Audit duplicates, resolution, compression,
metadata, and masks.

### Recommended sample sizes

- Presentation pilot: 1,000–2,000 image instances and 50–100 independent groups.
- Final undergraduate study: 10,000–20,000 instances and ≥300 independent groups.
- External test: at least 100–300 images kept untouched until final reporting.

## 5. Evaluation protocol

Primary metrics are ROC-AUC and PR-AUC. Report balanced accuracy, precision,
recall, F1, confusion matrix, and per-manipulation recall. Add calibration, TPR at
fixed FPR, confidence intervals, three random seeds, inference time, and memory in
the final paper.

Do not tune thresholds on the test set. Do not split variants from one source into
different folds. Report the pilot’s synthetic-data limitation next to every chart.

## 6. Protection and audit protocol

For each input, record filename, SHA-256, model version, score, decision, and UTC
timestamp. A reference hash can later be used to detect byte-level changes. AES-GCM
is used only as an optional local confidentiality demonstration and the key stays
outside the repository.

## 7. Next experiments

1. Add CASIA v2.0 and CoMoFoD small with verified licenses and grouped splits.
2. Add a majority baseline, random forest, and a small CNN/ResNet comparison.
3. Add JPEG quality 50/70/90, resize, crop, and blur robustness sweeps.
4. Add localization masks and report IoU/Dice only on datasets with valid masks.
5. Repeat with three seeds and bootstrap confidence intervals.
