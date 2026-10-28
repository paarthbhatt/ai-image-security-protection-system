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

Known issue to disclose: a documented share of CASIA labels and masks is wrong.
Cite the corrections repo when you report localisation quality.

## CoMoFoD (small)

Direct download, no registration: <https://www.vcl.fer.hr/comofod/>

Take `CoMoFoD_small_v2.7z` (200 base scenes, 10,400 images including
post-processed variants). The loader recognises `<scene>`, `<scene>_F` and
`<scene>_M` naming. **Count 200 scenes, not 10,400 images** — the variants are
not independent samples, and treating them as independent inflates every metric.

`comofod_small.rar` is 2.99 GiB, the server throttles to roughly 100 KB/s, and
that works out to about 9 hours. `7z` is not installed on this machine; `bsdtar`
is, and its libarchive build reads `.rar` archives.

## COVERAGE

OneDrive share, browser download only — the GitHub repository
(<https://github.com/wenbihan/coverage>) contains nothing but a README, and the
share returns HTTP 403 to any script, so `curl` cannot fetch it.

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

# Synthetic benchmark on natural images, three seeds
python src/run_pipeline.py --dataset cifar10 --source-images 5000 --seeds 42,7,13
```

## Rules the loader already enforces

- **Grouped splits.** Every variant of one scene stays in one fold.
- **No silent substitution.** A missing dataset raises `DatasetUnavailable`; the
  digits fallback requires an explicit `--allow-fallback`.
- **Provenance in the data.** Each row carries its corpus name and a source id
  derived from the file that produced it.
- **Masks are kept.** `mask_path` is populated when a ground-truth mask is found.

## What is still missing before claiming "authentic"

1. **An untouched external test set.** Train on CASIA v2.0, report COVERAGE and
   CoMoFoD without refitting. Cross-dataset drop is the honest headline number;
   in-distribution accuracy is not evidence of generalisation.
2. **Localisation metrics.** Masks are stored but nothing scores IoU/Dice yet.
   Adding a localisation head is the next real step up in rigour.
3. **A deep baseline.** The current detector is handcrafted-feature based. A
   small constrained-convolution CNN (Bayar & Stamm, IH&MMSec 2016) or
   TruFor/Mesorch-style model is the literature comparison reviewers expect.
4. **Calibration and TPR-at-fixed-FPR.** A screening tool is judged by its
   false-positive rate at a usable operating point.
