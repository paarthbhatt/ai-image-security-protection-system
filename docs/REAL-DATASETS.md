# Using real forgery datasets

The synthetic benchmark answers "can these features separate *our* edits?"
Real benchmarks answer "do they separate *human* forgeries?" Only the second
question is publishable, so the pipeline supports both and never mixes them.

## Expected directory layout

Put each dataset under `data/real/<name>/` and pass `--real-dataset <name>`.
The loader in `src/sources.py` auto-detects folder roles from the folder names
(`Au`, `Tp`, `authentic`, `forged`, `mask`, `_F`, `_M`, ...), then splits by
scene so no scene leaks across folds.

```
data/real/
├── casia2/
│   ├── Au/            # authentic images
│   └── Tp/            # tampered images
│   └── CASIA 2 Groundtruth/   # optional masks
├── comofod/
│   ├── 1_original.png         # authentic
│   ├── 1_F.png                # forged
│   └── 1_M.png                # mask
└── coverage/
    ├── image/                 # authentic + forged pairs
    └── mask/                  # ground-truth masks
```

If your download uses different folder names, either rename them to one of the
hints above or extend `AUTHENTIC_HINTS` / `TAMPERED_HINTS` / `MASK_HINTS` in
`src/sources.py`.

## CASIA v2.0

The canonical distribution (`forensics.idealtest.org`) is registration-only and
intermittently offline, so it usually has to be fetched by hand.

| Route | Notes |
|---|---|
| Kaggle `divg07/casia-20-image-tampering-detection-dataset` | 2.56 GB. Needs a Kaggle account and the Kaggle CLI (`kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset`). Unpacks to `CASIA2/Au` and `CASIA2/Tp`. |
| `github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth` | Maintained corrections for the known label/mask errors, plus a mirror link. Use this if you report mask-based metrics. |

Approved size: 7,491 authentic and 5,123 tampered images (CASIA v2.0). Report
**scene counts, not file counts** — several tampered files derive from one scene.

### The authentic set is not independent of the tampered set

CASIA names each forgery after the photographs it was built from:

```
Au_ani_00018.jpg                                   the photograph
Tp_D_CND_M_N_ani00018_sec00096_00138.tif           that photograph, patched
```

So `Au_ani_00018.jpg` and every `Tp_*_ani00018_*` are the same scene, one
untouched and one edited. Keyed by filename they land in different folds, and the
model then sees identical content labelled authentic in one split and tampered in
another — leakage that *inflates* the score instead of degrading it.

The linkage has to be transitive. CASIA reuses a small pool of photographs, and
3,020 of its 5,123 forgeries trace back to 796 shared originals through chains of
overlapping source ids. Measured on the 7,491 + 5,123 layout this gives 6,523
scenes, of which one holds 3,814 images (794 authentic, 3,020 tampered) — 30% of
the dataset. That scene cannot be divided, so whichever fold holds it carries a
higher forged rate than the others. It is a property of the benchmark, not a
splitter bug, and it has to be disclosed with any per-fold numbers.

Known issue to disclose: a documented share of CASIA labels and masks is wrong.
Cite the corrections repo when you report localisation quality.

## CoMoFoD (small)

Direct download, no registration: <https://www.vcl.fer.hr/comofod/>

The one file the server actually serves is
`https://www.vcl.fer.hr/comofod/comofod_small.rar` — **3,064 MiB**. Measured on
2026-11-06 at **1.6–2.0 MiB/s**, so about **25 minutes**, not the nine hours an
earlier note here predicted (that estimate assumed 100 KB/s, which the server no
longer imposes). `CoMoFoD_small_v2.7z`, `CoMoFoD.rar` and `CoMoFoD.zip` all
return 404 — do not chase them. `.7z` tooling is not needed and `7z` is not
installed; `bsdtar` reads this RAR.

Fetch it with `python .work/fetch_comofod.py` (resumes with `Range`, refuses to
rename a truncated archive, and writes a `.sha256` sidecar — a dataset whose
bits cannot be named cannot be cited), then audit it with
`python .work/unpack_comofod.py`.

Five manipulation geometries — translation, rotation, scaling, distortion,
combination — each with post-processing variants (JPEG, blur, noise, colour
reduction). The loader recognises `<scene>`, `<scene>_F` and `<scene>_M` naming.

**Count scenes, not files** — the variants are not independent samples, and
treating them as independent inflates every metric.

**Composition as measured in our copy**, by `.work/audit_comofod_loader.py` and a
file-count check of the archive: scene ids run contiguously 001–200, every scene
ships all 52 files (1 `_B`, 25 `_O`, 25 `_F`, 1 `_M`), so 10,400 images in total
and 10,200 labelled rows once `_B` is excluded — 5,200 authentic, 5,000 forged,
every forgery carrying a mask. This is a complete archive, not a truncated one.

An earlier note here said "260 forged image sets". Nothing in this copy supports
that number and nobody has traced where it came from, so it is treated as
unverified rather than corrected: cite 200 scenes, which is counted, and leave
260 out of the submission until someone finds it in writing.

`readme.txt` inside the archive assigns every scene to one geometry section
(translation 40, rotation 40, scaling 40, distortion 39, combination 41), and
`src/sources.py` reads that assignment into `tamper_type`, so transfer can be
reported per geometry instead of pooled. The families are a property of the
corpus rather than an invention of ours, which is what made the per-geometry
breakdown in [docs/TRANSFER-FINDING.md](TRANSFER-FINDING.md) falsifiable — and it
was falsified: all five geometries transfer at chance.

## Columbia image splicing set

**Not scriptable.** `ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/`
returns HTTP 403 to every non-browser request, and the archive sits behind a
terms-acceptance form (`dlform.html`). 20.69 MB in RAR, 183 images (69 authentic,
112 spliced) plus masks. `bsdtar` reads RAR, so once a browser has downloaded it,
unpacking is: `bsdtar -xf <archive> -C data/real/columbia`. Registered as
`--real-dataset columbia` / `--cross-dataset columbia`; the loader maps
`AuthenticImages`/`TamperedImages` onto the two classes and labels every
Columbia forgery `splicing`, which is how the corpus is constructed.

## COVERAGE

OneDrive share, browser download only — the GitHub repository
(<https://github.com/wenbihan/coverage>) contains nothing but a README, and the
share returns HTTP 403 to any script, so `curl` cannot fetch it. Re-checked
2026-11-06: still true.

| Route | Notes |
|---|---|
| <https://1drv.ms/f/s!AggVhXcCj1FLhUUyUrqSpV_yI_GH> | Open in a browser, download the folder. Works without an account. |
| <https://pan.baidu.com/s/11i_swrFveLc9uZr1eR006Q> | Baidu mirror, password `zduj`, for mainland China. |

100 source images, each with a genuine and a copy-move forged version, plus
masks. Because there are only 100 groups, treat COVERAGE as the *external* test
set rather than a training set.

## Other useful benchmarks

| Dataset | Size | Use |
|---|---:|---|
| DSO-1 | 100 scenes | Splicing, compact external check |
| MICC-F220 / F2000 | 220 / 2000 | Copy-move with geometric transforms |
| Realistic Tampering (Korus) | 440 | Realistic, camera-model metadata |
| IMD2020 / Defacto | ~35k / ~5k | Large-scale, deep-learning territory |

## Running

```bash
# Real benchmark, no synthesis at all
python src/run_pipeline.py --real-dataset casia2

# In-domain metrics, region localisation (Dice/IoU), then zero-shot transfer to
# two corpora the model has never seen. This is the publishable run.
python src/run_pipeline.py --real-dataset casia2 \
    --cross-dataset comofod,coverage --localise-limit 240

# Synthetic benchmark on natural images, three seeds
python src/run_pipeline.py --dataset cifar10 --source-images 5000 --seeds 42,7,13
```

`--localise-limit 0` skips the tile pass, which is the slowest step in the run.
Extra seeds report image-level metrics only; the tile pass does not depend on
the split seed, so repeating it there would buy nothing and cost an hour.

## Rules the loader already enforces

- **Grouped splits.** Every variant of one scene stays in one fold.
- **No silent substitution.** A missing dataset raises `DatasetUnavailable`; the
  digits fallback requires an explicit `--allow-fallback`.
- **Provenance in the data.** Each row carries its corpus name and a source id
  derived from the file that produced it.
- **Masks are kept.** `mask_path` is populated when a ground-truth mask is found.

## What is still missing before claiming "authentic"

1. **An untouched external test set.** *Code path built and verified
   (`src/cross_dataset.py`, `--cross-dataset`): model, decision threshold and
   region thresholds are all inherited from the training corpus, so the external
   data tunes nothing. Still needs the corpora themselves downloaded.*
   Cross-dataset drop is the honest headline number; in-distribution accuracy is
   not evidence of generalisation.
2. **Localisation metrics.** *Done for the classical detectors:* tiled inference
   plus an ELA reference, scored as Dice/IoU/hit-rate against the stored masks,
   with region thresholds chosen on validation only (`src/localise.py`). On
   CASIA the numbers are weak (tiled Dice around 0.07–0.12), which is a result
   about whole-image classifiers on crops, not a measurement bug. A segmentation
   network is still missing.
3. **A deep baseline.** The current detector is handcrafted-feature based. A
   small constrained-convolution CNN (Bayar & Stamm, IH&MMSec 2016) or
   TruFor/Mesorch-style model is the literature comparison reviewers expect.
   CUDA PyTorch is installed and verified on the RTX 2050.
4. **Calibration and TPR-at-fixed-FPR.** A screening tool is judged by its
   false-positive rate at a usable operating point.

## Sources considered and rejected

HuggingFace carries copy-move mirrors (`TheAnalyzer/CopyMoveForgeryDataset`,
`Emulated-Inc/copymove`, `lorenzo-morelli/image-splicing-deepfake-mix`), all
fetchable without a token. They were rejected on provenance: none is the
published CoMoFoD, Columbia or COVERAGE, none carries the authors' masks, and a
forensics reviewer checking the dataset name against the citation would find a
mismatch. Using them would buy a second number at the cost of the claim.
