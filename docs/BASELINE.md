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

Test-set metrics from the current CASIA run. MCC is derived from the stored
confusion matrix; it is not yet emitted by the pipeline.

| | Paper ResNet-CNN | Ours LR | Ours RF | Ours HGB |
|---|---:|---:|---:|---:|
| ACC | 0.8851 | 0.7979 | 0.8765 | **0.8786** |
| P | 0.8123 | 0.7036 | 0.8249 | 0.8162 |
| R | 0.9415 | 0.8282 | 0.8655 | 0.8868 |
| F1 | 0.8664 | 0.7609 | 0.8447 | **0.8500** |
| AUC | 0.9596 | 0.8756 | 0.9416 | **0.9472** |
| MCC | 0.7744 | 0.5938 | 0.7429 | **0.7501** |

We are within a few points on every metric with a handcrafted-feature model and
no deep network. Two things stop this being a claim we can make yet:

1. Their CASIA is **modified** (see protocol below) and ours is not, so our 0.9472
   is still carrying the file-format shortcut they designed out.
2. Their set is class-balanced and ours is not, so accuracy is not comparable
   until we balance.

After normalising format ourselves — with a fixed q90 rather than their random
quality — our AUC fell to 0.868, which is below them. That is the honest gap to
close.

A fairness caveat that belongs in the report, not in a footnote: we split by
source scene and they state no scene grouping. If their split was random at the
file level, CASIA forgeries share source photographs across train and test, so
their reported numbers are inflated relative to ours in a way that flatters us
once the shortcut is removed. Say so plainly rather than letting the table imply
a closer comparison than the protocols allow.

### Their own ablation supports our finding

Table 8 of the paper, each branch of their two-branch model trained alone:

| Branch | ACC | F1 | AUC | MCC |
|---|---:|---:|---:|---:|
| Spatial only | 0.6957 | 0.6787 | 0.7602 | 0.3922 |
| Compression-artefact only | 0.7966 | 0.7773 | 0.8718 | 0.5971 |
| Both combined | 0.9382 | 0.9378 | 0.9802 | 0.8772 |

Neither branch is strong alone, and the compression branch clearly outperforms
the spatial branch. This is the same pattern our ablation found: forensic
residual/copy-move evidence alone supports 0.671, compression-shaped features
alone support 0.903. Cite this — it means our analysis reproduces a published
result rather than being a curious observation of our own.

## Shared protocol — all four members must follow this

Differences here are what make cross-paper comparisons meaningless. This is
specified so the four models are comparable to each other and to the paper.

### Dataset

- Source: CASIA v2.0, `data/real/casia2/` — 7,491 authentic (`Au/`), 5,123
  tampered (`Tp/`), 5,123 ground-truth masks (`Gt/`).
- **Format normalisation, matching the paper:** every tampered image stored
  uncompressed (TIFF/BMP — 3,059 of 5,123 are TIFF) is re-encoded as JPEG at a
  **random quality factor**. Their stated reason is that their framework uses
  compression artefacts, so leaving images uncompressed makes the file format,
  not the forgery, the discriminative feature.
  - Use a fixed random seed so all four members get byte-identical datasets.
  - A fixed quality for everything is *not* equivalent: it leaves originally
    JPEG images double-compressed and originally TIFF images single-compressed,
    which is still a class-correlated cue. Randomising quality is what removes
    it.
- **Class balance:** subsample authentic to 5,123, matching the paper. Report
  class counts in every results table.
- Expected total: **10,246 images**.

### Splits

- Grouped by scene, never by file. CASIA names each forgery after the
  photographs it was pasted from, so `Au_ani_00018.jpg` and every
  `Tp_*_ani00018_*` are one scene; the linkage is transitive.
  12,614 images resolve to 6,523 scenes; the balanced set will be fewer.
- 70 / 15 / 15 train / validation / test, assigned by whole scene.
- Splits are **frozen**. One member produces `benchmark.csv` and the metadata
  sidecar; the other three consume it unchanged. Nobody re-splits.
- A run is invalid if the group-leakage check is nonzero.

### Metrics

Report all six, in this order, to match the paper's tables:

`ACC  P  R  F1  AUC  MCC`

`matthews_corrcoef` is the only one not currently emitted by the pipeline and
needs adding. Its value for the current model is 0.7501, computed from the stored
confusion matrix as a stopgap. Threshold is chosen on validation only, never on
test. Report the threshold used.

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

Beating this baseline is enough for the course, and enough for the extra marks.
It is not by itself enough to submit a paper. Reviewers in this field expect a
second dataset (CoMoFoD, COVERAGE, or PROV+) to show the result is not CASIA-
specific, and localisation metrics (IoU or Dice) rather than image-level
accuracy alone. We already load and render the ground-truth masks, so Dice and
IoU are the cheaper of the two gaps; a second dataset is the expensive one.
Decide which of those the group is aiming at before anyone starts a long
training run.
