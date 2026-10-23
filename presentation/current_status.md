# Current Status Presentation — AI-Based Image Security and Protection System

## 1. One-sentence update

We have built and executed a reproducible baseline that downloads image data,
creates source-safe tamper examples, trains an interpretable forensic detector,
checks cryptographic protection, and generates visual evidence.

## 2. What we can show today

- **Dataset:** 1,000 image instances from
  500 source groups (scikit-learn digits fallback (8x8 grayscale, upscaled to 32x32 RGB)); balanced
  authentic/manipulated labels.
- **Manipulations:** copy-move, splicing, brightness manipulation, and JPEG
  recompression.
- **Model:** standardised handcrafted forensic features + balanced logistic
  regression; 22 features.
- **Held-out test:** ROC-AUC **0.939**, PR-AUC **0.952**,
  balanced accuracy **0.873**, F1 **0.871**.
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
5. **Honesty:** this pilot uses controlled synthetic edits on small images; it is
   a working baseline, not proof of authenticity.
6. **Next step:** external datasets (CASIA v2.0, CoMoFoD small, and an untouched
   external test set), a CNN comparison, and robustness/calibration analysis.

## 4. Questions the supervisor may ask

**Why not report accuracy only?** Accuracy can hide class imbalance and threshold
effects, so we report ROC-AUC, PR-AUC, balanced accuracy, F1, and a confusion
matrix.

**How did you avoid leakage?** We split source images into train/validation/test
before generating manipulated counterparts. Variants from the same source stay in
one split.

**Does hashing detect a visual edit?** Only if a trusted original hash exists and
the bytes changed. It complements—not replaces—the AI detector.

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
- `figures/robustness.png` — post-processing degradation.
- `reports/results.json` — machine-readable metrics.
- `docs_research_plan.md` — research questions, threat model, data protocol.
