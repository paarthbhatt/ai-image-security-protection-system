# Baseline Paper, Target Numbers, and Shared Protocol

This is the reference document for the four-model study. Every member trains
against the same data and reports the same six metrics in the same order.

## Base research paper

### Published citation — use this one

A. Yadav and D. K. Vishwakarma, "Toward effective image forensics via a novel
computationally efficient framework and a new image splice dataset,"
*Signal, Image and Video Processing*, vol. 18, no. 4, pp. 3341-3352, June 2024.
DOI: 10.1007/s11760-024-02997-6 (online 16 Mar 2024).

Note the published title says **"Toward"**; the arXiv preprint says **"Towards"**.
Cite the journal form. The preprint is
<https://arxiv.org/abs/2401.06998>, and a copy of that PDF is in
`docs/reference/` because everything below was read from it.

Venue standing, verified against Crossref and journal metrics rather than
assumed: SCIE and Scopus indexed, SJR Q2, H-index 66, impact factor around 2.0-2.7,
3 citations on Crossref / 7 on Google Scholar as of this writing. A solid,
mid-tier, peer-reviewed application paper — not a flagship. Describe it to the
faculty as *a recent published baseline on our exact dataset*, never as *the
state of the art*, which it is not.

Supporting reference to cite alongside it, from the same group and a Q1 venue:
A. Yadav, D. Gupta and D. K. Vishwakarma, "Datasets, clues and state-of-the-arts
for multimedia forensics: an extensive review," *Expert Systems with
Applications*, vol. 249C, 2023. It backs the dataset and feature-family choices
so the study does not rest on a single mid-tier paper.

Chosen because it uses the same dataset (CASIA v2.0), the same task (image-level
authentic-vs-tampered classification), the same metric suite, and it reports a
comparison table of prior work on CASIA that we can place our row into directly.

### What the paper actually reports

The abstract's headline — **ACC 0.9382, F1 0.9378, AUC 0.9802, MCC 0.8772** — is
**not a CASIA result**. It is Table 7, trained and tested on **AMC**, their
32,770-image combined set (two newly proposed splice datasets plus modified
CASIA v2.0). It must not be quoted as a CASIA number.

Their CASIA-only result is Table 9. Every number in this document came from the
**arXiv** PDF; the journal version is paywalled. They should be identical, but
the journal governs, so spot-check Table 9 through the college library login
before quoting 0.9596 as *their* number in a submitted report.

| Model | ACC | P | R | F1 | AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| (Proposed) ResNet-CNN | 0.8851 | 0.8123 | 0.9415 | 0.8664 | **0.9596** | 0.7744 |
| (Proposed) DenseNet-CNN | 0.8125 | 0.7823 | 0.7646 | 0.7733 | 0.9126 | 0.6136 |

Other rows in their Table 9, all F1 unless noted — the published field on CASIA:

| Method | F1 | Notes |
|---|---:|---|
| Zhang et al. | 0.7653 | best prior F1 in their table |
| Yan et al. | 0.7350 | P 0.8090, R 0.7460 |
| Chen et al. | 0.7388 | |
| Chen et al. | 0.7051 | P 0.6616, R 0.7548 |
| Salloum et al. | 0.6675 | |
| Wu et al. | 0.6510 | P 0.6310, R 0.6730 |
| Bi et al. | 0.5860 | P 0.6780, R 0.5860 |
| Wu et al. | 0.5770 | MCC 0.5590 |
| Zhou et al. | 0.5709 | P 0.5044, R 0.6575 |
| Liu et al. | 0.5232 | |
| Xu et al. | 0.4601 | AUC 0.8191 |

**The target to beat is their ResNet-CNN row: ACC 0.8851 / F1 0.8664 /
AUC 0.9596 / MCC 0.7744.**

### Where we stand today

Test-set metrics from the current CASIA run (2026-11-06, feature version
`2026-11-dct-block-copymove`). All six columns are emitted by the pipeline
itself — MCC is computed in `_metrics` alongside them, not derived afterwards.

This table and the paper's row are **not** like-for-like: the corpus under these
numbers is CASIA as distributed, with the format shortcut intact. The like-for-like
comparison is the modified-CASIA table two paragraphs below.

| | Paper ResNet-CNN | Ours LR | Ours RF | Ours HGB |
|---|---:|---:|---:|---:|
| ACC | 0.8851 | 0.8021 | 0.8677 | **0.8718** |
| P | 0.8123 | 0.7105 | 0.7822 | 0.7976 |
| R | 0.9415 | 0.8269 | 0.9134 | 0.8975 |
| F1 | 0.8664 | 0.7643 | 0.8428 | **0.8446** |
| AUC | 0.9596 | 0.8744 | 0.9394 | **0.9430** |
| MCC | 0.7744 | 0.6005 | 0.7363 | **0.7399** |

The three "ours" columns are the same three models the pipeline fits; HGB is the
selected one. The previous run of this table read AUC 0.9472 / MCC 0.7501. The
0.004 AUC difference is the repaired copy-move group: eight columns that used to
be identically zero now vary, and in a set already dominated by compression
statistics they dilute the boosting model slightly. It is reported rather than
undone by deleting the group, because the same eight columns are worth +0.037
AUC in the forensic-only arm below. See the paired measurement in
`docs/TRANSFER-FINDING.md`.

We are within a few points on every metric with a handcrafted-feature model and
no deep network. Two things stopped this being a claim we could make:

1. Their CASIA is **modified** — every uncompressed image re-encoded at a random
   JPEG quality factor — and ours was not, so our 0.9430 carried the file-format
   shortcut they designed out.
2. Their set is class-balanced and ours was not, so accuracy was not comparable.

Both are now closed by running their protocol rather than arguing about it.
`src/modified_casia.py` builds the corpus §4.1.3 describes (3,103 conversions,
quality factors 50–100 drawn per image from a hash, authentic thinned to 5,123,
10,246 rows, manifest digest in `reports/modified_casia_build.json`) and the
identical pipeline ran on it under `--arm casia2_mod`:

| | Paper ResNet-CNN | Ours LR | Ours RF | Ours HGB |
|---|---:|---:|---:|---:|
| ACC | 0.8851 | 0.7595 | **0.8592** | 0.8232 |
| P | 0.8123 | 0.7252 | **0.8071** | 0.7751 |
| R | 0.9415 | 0.7866 | **0.9208** | 0.8792 |
| F1 | 0.8664 | 0.7547 | **0.8602** | 0.8239 |
| AUC | 0.9596 | 0.8113 | **0.9211** | 0.8974 |
| MCC | 0.7744 | 0.5212 | **0.7260** | 0.6530 |

Same six columns, same corpus shape as theirs, 7,100 / 1,562 / 1,584 split, group
leakage check passes (`groups_in_multiple_splits: 0`). Random forest is the
selected detector here — on this corpus it beats histogram gradient boosting on
validation AUC, which is the rule, not a preference.

The gap that remains is small and it is not uniform: precision is within 0.005,
F1 within 0.006, accuracy within 0.026, while AUC is 0.039 behind and MCC 0.048.
Read the AUC and MCC columns as the honest distance: their deep two-branch model
separates the classes more cleanly at the top of the ranking than 143 handcrafted
statistics do, even though at one threshold the two are nearly indistinguishable.

What is still not the same, and each of these would move a number:

- **Split grouping.** We place whole scenes in one fold; the paper states no
  grouping. CASIA's scenes are violently uneven — the median scene is a single
  image, but one scene (`ani00001`) holds 3,814 rows including 3,020 of the 5,123
  tampered images, 58.9% of the whole positive class. A file-level random split
  can therefore put an original photograph in train and dozens of splices made
  from it in test, and on that scene it is not an edge case but the majority of
  the positives. If they split at file level their numbers are inflated relative
  to ours, which means this comparison flatters us; the paper does not say, and
  the honest position is that we cannot tell.
- **Quality-factor range.** §4.1.3 says "random quality factor" and never gives
  the range. 50–100 is our choice, it is recorded in `reports/modified_casia_build.json`
  (mean 74.58), and it is the single most likely source of a disagreement with
  their numbers that no one will be able to resolve from the text.
- **Augmentation.** Theirs is horizontal flip, vertical flip and rotation through
  180°. Ours keeps the flips and drops rotation, because resampling pixels is
  precisely the evidence a noise and DCT detector is asked to read. Our training
  half additionally sees six post-processed copies of itself, which is not in
  their protocol.
- **Machine.** They report 128 GB RAM and a 24 GB TITAN RTX. This arm ran on a
  15 GiB, 16-core laptop, and the classical pipeline does not use its GPU at all.
  That is not a modesty note: it is why the feature model is the part that is
  reproducible here, and why the deep baseline in `src/deep_detector.py` reports
  its peak VRAM instead of assuming the reader has the same card.

The parity arm is also the first place the robustness sweep means something new.
Every condition scores lower on it than on the distributed corpus — the shortcut
is gone and every image has already been through one JPEG generation — but the
*drop from clean* is not uniformly worse: a further re-encode at q90 costs the
parity corpus 0.067 AUC where it costs the distributed corpus 0.110, exactly as
you would expect once the training set has already seen random-quality
re-compression. Re-scaling to half resolution behaves oppositely (0.187 against
0.145). Two caveats on reading these numbers as a comparison: the selected
detectors differ between the arms (random forest here, gradient boosting there),
so this is not a controlled A/B, and the clean rows themselves are not equal.

Table 9 itself is ambiguous about its own provenance: the text says every
architecture was retrained "instead of merely comparing with metric figures
mentioned in published works", while its footnote says "-" marks scores "not
mentioned in the research papers", which implies at least the competitor rows are
quoted from other work.

The earlier fixed-q90 probe (AUC 0.873, 400 test images per class, re-measured
with the repaired copy-move group; 0.868 under the previous feature version) is
superseded for this purpose by the arm above, which is the full corpus and their
protocol rather than a subsample with a hand-picked quality factor. It stays in
`docs/TRANSFER-FINDING.md` because the q90 probe answers a different question —
how much of the *shortcut* a single codec state removes.

### Their own ablation supports our finding

Table 8 of the paper, each branch of their two-branch model trained alone:

| Branch | ACC | F1 | AUC | MCC |
|---|---:|---:|---:|---:|
| Spatial only | 0.6957 | 0.6787 | 0.7602 | 0.3922 |
| Compression-artefact only | 0.7966 | 0.7773 | 0.8718 | 0.5971 |
| Both combined | 0.9382 | 0.9378 | 0.9802 | 0.8772 |

Neither branch is strong alone, and the compression branch clearly outperforms
the spatial branch. This is the same pattern our ablation found: forensic
residual/noise/copy-move evidence alone supports 0.698, compression-shaped
features alone support 0.887. Cite this — it means our analysis reproduces a
published result rather than being a curious observation of our own. Both numbers
are from the current feature version; the 0.671 this document used to quote was
measured while the copy-move group was silently returning zeros, which is why
the spatial branch looked weaker than it is.

## Shared protocol — all four members must follow this

Differences here are what make cross-paper comparisons meaningless. This is
specified so the four models are comparable to each other and to the paper.

### Dataset

- Source: CASIA v2.0, `data/real/casia2/` — 7,491 authentic (`Au/`), 5,123
  tampered (`Tp/`), 5,123 ground-truth masks (`Gt/`).
- **Format normalisation, matching the paper — built, in `src/modified_casia.py`:**
  every uncompressed image is re-encoded as JPEG at a **random quality factor**,
  and the authentic class is thinned to 5,123. Measured on our archive that is
  3,059 tampered TIFFs plus 44 of the 54 authentic BMPs — 3,103 conversions —
  giving 5,123/5,123 and **10,246 rows**, 4,781 scenes.
  - Both classes are converted. The paper's sentence reads as if only tampered
    images are touched, and `--tampered-only` reproduces that literal reading,
    but converting only the tampered side would make "this file is a BMP" a
    pure authentic-class cue. The arm exists to remove cues, not relocate them.
  - The quality factor is drawn from a hash of `seed:path`, not from a shuffle
    over the list, so adding or removing one image cannot reshuffle everyone
    else's draw. Range 50–100 (mean 74.58); the paper never states a range.
  - A fixed quality for everything is *not* equivalent: it leaves originally
    JPEG images double-compressed and originally TIFF images single-compressed,
    which is still a class-correlated cue. Randomising quality is what removes
    it.
  - The corpus is 9 GB and cannot be committed, so the shared artifact is the
    recipe plus its digest: `reports/modified_casia_build.json` records the
    manifest SHA-256, and `.venv/bin/python src/modified_casia.py verify` re-checks
    dimensions and that digest on any machine that has the CASIA archive.
  - Every converted row carries the `source_id` and `tamper_type` of the image it
    came from. Re-deriving them from the new path would break CASIA's scene
    linkage and, with it, the leakage rule below — silently, and only on this arm.

### Splits

- Grouped by scene, never by file. CASIA names each forgery after the
  photographs it was pasted from, so `Au_ani_00018.jpg` and every
  `Tp_*_ani00018_*` are one scene; the linkage is transitive.
  12,614 images resolve to 6,523 scenes; the balanced set has 10,246 rows in
  4,781 scenes.
- 70 / 15 / 15 train / validation / test, assigned by whole scene. On the
  modified corpus that is 7,100 / 1,562 / 1,584 over 4,781 groups; on CASIA as
  distributed, 8,739 / 1,940 / 1,935 over 6,523.
- Splits are **frozen**. One command produces the benchmark CSV and the metadata
  sidecar; every other run consumes it unchanged, and `--arm <name>` keeps a
  side experiment's artifacts out of the cited ones instead of overwriting them.
- A run is invalid if the group-leakage check is nonzero.

### Metrics

Report all six, in this order, to match the paper's tables:

`ACC  P  R  F1  AUC  MCC`

`matthews_corrcoef` is emitted by the pipeline itself — it is computed in
`_metrics` in `src/train.py` beside the other five, so no metric in this table
has a different provenance from any other. Its value for the current model is
0.7399. Threshold is chosen on validation only, never on test. Report the
threshold used.

### Also reported, because it is our differentiator

The paper is in-domain only. Every member additionally reports the nine-condition
robustness sweep (clean, JPEG q90/q70/q50/q30/q20, half resolution, blur, noise).
This is the axis on which we have a real advantage to show, and it is the reason
a reviewer would keep reading.

## The four models

| Member | Model | Input | Owns |
|---|---|---|---|
| 1 | Logistic regression | 143 forensic features | Interpretable baseline; coefficient and feature-group analysis |
| 2 | Histogram gradient boosting | 143 forensic features | Our current best; feature ablation; robustness sweep |
| 3 | Fine-tuned pretrained CNN | 256×256 pixels | Pixel-learning track; the likely winner on AUC |
| 4 | **Hybrid** — CNN on pixels + 143 forensic features | both | The research contribution; direct competitor to the paper's two-branch design |

Member 4 is the one with a paper in it. The base paper's second branch exists to
capture compression artefacts, and our 143 features already compute exactly that
— so a hybrid tests whether handcrafted forensic features still add anything on
top of learned ones. That is a real question with a measurable answer either way.

## Constraint worth knowing early

The GPU is an RTX 2050 with **4 GB VRAM**. Deep members must fine-tune a small
pretrained backbone (ResNet-18 or EfficientNet-B0) at 224 or 256 with small
batch size and mixed precision. Their paper uses models with 19,510 to 99,538
trainable parameters, which is a useful reminder that a large model is not
required to compete here.

## Open questions to resolve before we start training

- The paper's exact random-quality distribution for the modified CASIA is not
  stated in the sections we read. Confirm from the full text, or specify our own
  range and report it.
- Whether they used their stated 5,123-of-7,491 authentic subsample for Table 9
  specifically is not explicit. Ours will be explicit.
- The corrected ground-truth release documents mask/image resolution mismatches,
  which matters as soon as anyone computes localisation metrics.

## What a journal reviewer would still ask for

Both gaps this document used to name are now closed, and closing them produced
the project's most important result:

- **A second dataset.** CoMoFoD small v2 is loaded, audited against its own
  published composition, and scored zero-shot. The detector is at chance there
  (ROC-AUC 0.506, MCC −0.003), and the mechanism is a post-processing confound
  rather than a labelling bug — see
  [docs/TRANSFER-FINDING.md](TRANSFER-FINDING.md).
- **Localisation metrics.** Dice, IoU, hit rate, pixel accuracy and the
  falsely-flagged area are reported for tile rescoring and for ELA, in-domain and
  per external corpus: in-domain tiled Dice 0.131, ELA 0.156, transfer Dice
  0.051. Weak numbers, measured honestly.

What reviewers would still ask for, in the order that costs us most:

1. The paper's **modified-CASIA protocol** (random quality factor per image,
   balanced 5,123/5,123), without which the comparison stays approximate.
2. **More external corpora** — one external dataset is an anecdote, and the base
   paper reports one corpus too, which is a gap in the literature rather than a
   flaw specific to us. Columbia and COVERAGE are blocked on downloads that need
   a human browser session.
3. **A deep baseline**, and a segmentation network that predicts the region
   instead of rescoring crops with a whole-image classifier.
4. **Multi-seed variance.** Everything here is seed 42.
