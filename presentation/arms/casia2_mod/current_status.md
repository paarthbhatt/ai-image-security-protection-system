# Current Status Presentation — AI-Based Image Security and Protection System

## 1. One-sentence update

We have built and executed a reproducible baseline that loads the casia2_mod forgery benchmark distributed by its authors, trains an
interpretable forensic detector, checks cryptographic protection, and generates
visual evidence.

## 2. What we can show today

- **Dataset:** 10,246 image instances from
  4,781 source groups (CASIA v2.0 modified (uncompressed images recompressed, class-balanced)); balanced
  authentic/manipulated labels.
- **Manipulations:** copy_move, splicing.
- **Model:** random_forest over 143
  forensic features (selected on validation ROC-AUC from hist_gradient_boosting, logistic_regression, random_forest).
- **Held-out test:** ROC-AUC **0.921** (95% CI 0.907–0.934),
  PR-AUC **0.885**, balanced accuracy **0.863**, F1 **0.860**, MCC **0.726**.
- **Region localisation** (240 images per split, 128 px tiles at stride 64): learned tiles give Dice **0.112**, IoU **0.072**, hit rate **1.000**; ELA tiles give Dice **0.107**, IoU **0.066**. On authentic images the tiled predictor falsely flags **0.836** of the area. Weak region scores are a result, not a rounding error: a classifier fitted on whole photographs loses context a 128-pixel crop cannot keep.
- **Zero-shot transfer (never trained, never tuned on):** **comofod** bal-acc 0.492, ROC-AUC 0.514, MCC -0.036 (200 unseen scenes). The decision threshold 0.41 and the region thresholds are carried over from the training corpus unchanged.
  - **By tamper geometry** (each family scored against that corpus's authentic images): comofod — combination 0.52, distortion 0.53, rotation 0.53, scaling 0.50, translation 0.49.
- **Leakage check:** no source group appears in two splits — **True**.
- **Training augmentation:** 21,300 extra rows from
  jpeg_q90, jpeg_q70, jpeg_q50, jpeg_q30, noise, blur (never from test data).
- **Protection:** original hash match = **True**;
  mutated file rejected = **True**;
  AES-GCM round trip = **True**;
  modified ciphertext rejected = **True**.

## 3. Suggested five-minute talk track

1. **Problem:** images can be altered, and ordinary visual inspection is not
   reliable at scale.
2. **Threat model:** an attacker may copy, splice, recolour, or recompress an
   image; our system screens for manipulation and logs the evidence.
3. **Pipeline:** input → SHA-256/audit → forensic feature extraction → AI risk
   score → human-review decision → optional AES-GCM storage.
4. **Evidence:** show `figures/system_diagram.png`, then
   `figures/model_comparison.png`, `figures/confusion_matrix.png`, and the
   ROC/PR curves.
5. **Honesty:** the corpus is real tampered photographs and the splits are scene-grouped, so these numbers characterise this dataset rather than prove a universal detector; region localisation is reported even where it is weak.
6. **Next step:** zero-shot transfer to Columbia/COVERAGE and CoMoFoD, a fine-tuned CNN backbone, and a segmentation network that predicts the tampered region instead of describing it.

## 4. Questions the supervisor may ask

**Why not report accuracy only?** Accuracy can hide class imbalance and threshold
effects, so we report ROC-AUC, PR-AUC, balanced accuracy, F1, and a confusion
matrix.

**How did you avoid leakage?** We split source images into train/validation/test
before generating manipulated counterparts. Variants from the same source stay in
one split.

**Does hashing detect a visual edit?** Only if a trusted original hash exists and
the bytes changed. It complements—not replaces—the AI detector.

**How do you know the model localises rather than just detects?** Image-level
accuracy cannot tell those apart, so the tampered region is scored separately
against the ground-truth masks (Dice, IoU, hit rate, and false-positive area on
authentic images). Region thresholds are chosen on the validation split only.

**How many datasets are needed?** One public dataset is sufficient for a pilot;
the final study should use CASIA v2.0 + CoMoFoD small for development and CASIA
v1.0 or COVERAGE as an untouched external test. Target 10,000–20,000 instances
from at least 300 independent source groups.

## 5. Files to open during the presentation

- `reports/current_status.html` — visual dashboard.
- `figures/system_diagram.png` — architecture.
- `figures/sample_pairs.png` — authentic vs. manipulated examples.
- `figures/model_comparison.png` — baseline vs. trained detector.
- `figures/roc_pr_curves.png` — threshold-independent evidence.
- `figures/localisation.png` — where the tampering is, not just whether.
- `figures/robustness.png` — post-processing degradation.
- `reports/results.json` — machine-readable metrics.
- `docs_research_plan.md` — research questions, threat model, data protocol.
