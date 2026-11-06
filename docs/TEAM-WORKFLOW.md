# Who runs what, and how to run it

This exists because one member has the environment and the other three should
still be able to finish the project. It states what each person actually does,
the exact commands for the training runs, and the checks that decide whether a
number is allowed into the report.

Rules the whole document rests on:

* **Only committed evidence goes in the report.** Every figure and metric in
  `reports/results.json` is produced by one command, so nobody hand-adjusts a
  number, and any claim can be traced to the run that produced it.
* **Nothing about an external corpus may tune the detector.** See
  `docs/REAL-DATASETS.md` for how that is enforced in code.
* **A suspicious result is diagnosed before it is celebrated.** Two findings in
  this project came from refusing to accept a good-looking or a bad-looking
  number without asking why: the copy-move feature group that had been silently
  returning zeros since it was written, and the chance-level transfer score that
  turned out to be a post-processing confound rather than a labelling bug.

---

## Division of labour

| Member | Owns | Needs a terminal? |
|---|---|---|
| You | environment, all training runs, every number in the report, the figures | yes |
| B | literature and baseline verification: confirm the base paper's Table 9 numbers, its modified-CASIA protocol, and that our comparison quotes the right table | no |
| C | dataset acquisition and provenance: download the corpora that need a browser and a licence form, fill `data/real/<name>/SOURCE.txt`, count what arrived and check it against the published composition | mostly no |
| D | report and slides: `presentation/current_status.md` is the talk track; every claim in the write-up gets cross-checked against `reports/results.json` and the value recorded in the ledger below | no |

B and D work entirely inside the repository's documents. C's job is the part
that cannot be scripted — the Columbia terms form and the COVERAGE shared
password — plus the arithmetic that proves a download is complete rather than
partial.

### The evidence ledger (member D)

One row per number quoted in the report. It is what stops "0.95 accuracy" from
quietly becoming the wrong corpus's number.

Filled from the run of 2026-11-06 (feature version
`2026-11-dct-block-copymove`). Re-read these from `reports/results.json` rather
than copying them from this table; the table is the check, not the source.

| Claim in report | Source key in `reports/results.json` | Value | Date checked |
|---|---|---|---|
| In-domain ROC-AUC | `test.roc_auc` | 0.9430 (CI 0.933–0.953) | 2026-11-06 |
| In-domain MCC | `test.mcc` | 0.7399 | 2026-11-06 |
| Region localisation Dice | `localisation.tiled.dice` | 0.1313 (ELA 0.1564) | 2026-11-06 |
| Zero-shot transfer AUC | `cross_dataset.datasets.<name>.image_level.roc_auc` | 0.5062 (comofod) | 2026-11-06 |
| Zero-shot localisation Dice | `cross_dataset.datasets.<name>.localisation.tiled.dice` | 0.0510 (comofod) | 2026-11-06 |
| Transfer per tamper geometry | `cross_dataset.datasets.<name>.by_tamper_type.<family>.roc_auc` | 0.466–0.553, all at chance | 2026-11-06 |
| Robustness under recompression | `robustness.jpeg_q30.roc_auc` | 0.7291 | 2026-11-06 |

The conditions are direct keys of `robustness` (`robustness.clean`,
`robustness.jpeg_q30`, …), not rows of a `robustness.conditions` list. An earlier
copy of this table pointed at a key that does not exist, which is the kind of
error a ledger is supposed to catch before member D repeats it in the write-up.

Two traps this ledger exists to catch, both of them hit this project:

1. The base paper's abstract headline (**ACC 0.9382 / AUC 0.9802**) is *not* its
   CASIA result — that is Table 7 on a different dataset. Its CASIA row is Table
   9: **ACC 0.8851 / F1 0.8664 / AUC 0.9596 / MCC 0.7744**. Quote the second set.
2. `test` in our results is the in-domain held-out split. A transfer number lives
   under `cross_dataset`, and mixing the two would be the single most damaging
   error available in this report.

---

## Running the study

### Setup, once per machine

```bash
cd researchproject
python3 -m venv .venv
. .venv/bin/activate.fish          # fish; in bash: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is only needed for the deep track, and it is a separate install because
the CUDA wheels are large:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Verify the GPU is actually usable before believing any training time estimate:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On this machine that prints `2.15.0+cu132 True NVIDIA GeForce RTX 2050`.

### Datasets go here, and prove they are complete

```
data/real/casia2/       Au/  Tp/  Gt/          (never committed, ~3.4 GB)
data/real/comofod/      extracted/CoMoFoD_small_v2/  (+ readme.txt, SOURCE.txt)
data/real/coverage/     image/ mask/ authentic/
data/real/columbia/     AuthenticImages/ TamperedImages/ Mask/
```

Then assert the corpus matches its own publication instead of trusting a folder
listing:

```bash
.venv/bin/python .work/audit_comofod_loader.py
```

Expected: 10,200 rows, 5,200 authentic / 5,000 forged, 200 scenes, every forgery
masked, and every forgery labelled with the geometry its `readme.txt` documents.
That audit is what found the labelling defect that would otherwise have made
every CoMoFoD number meaningless.

### The one command that produces the paper's result set

```bash
.venv/bin/python -u src/run_pipeline.py \
    --real-dataset casia2 \
    --localise-limit 240 \
    --cross-dataset comofod \
    --cross-localise-limit 120 \
    --cross-variants 4
```

Measured on this machine: **11 m 49 s when the feature caches are warm** and
**33 m 25 s cold**, both completed 2026-11-06. Cold means `FEATURE_VERSION`
changed and all 12,614 images plus 26,220 augmented training copies, 9 robustness
conditions and 1,600 external images have to be re-extracted; extraction costs
about 0.2 s per image and dominates everything else. Two runs of the same command
on the same day took 11 m 49 s and 33 m 25 s — that ratio is the whole cost model
of this project.

What the flags mean, because the defaults are not the paper:

| Flag | Effect if you drop it |
|---|---|
| `--localise-limit 240` | `0` skips region localisation entirely, so no Dice, and the transfer run then refuses to score regions rather than tuning thresholds on external data |
| `--cross-dataset comofod` | nothing external is scored, and the report only has in-domain numbers |
| `--cross-variants 4` | scores all ~51 post-processing copies per photograph, inflating 200 photographs into a 10,200-sample benchmark |
| `--recompute-features` | unnecessary; the cache is validated against a content fingerprint and a version tag, and rebuilds itself when either changes |
| `--force` | rebuilds the benchmark from scratch; almost never what you want |

Use `python -u`. Without it the log buffers and the run looks frozen for
twenty minutes at a time. Do not pipe the run through `tail`/`head` for the same
reason — redirect to a file and read the file:

```bash
.venv/bin/python -u src/run_pipeline.py --real-dataset casia2 > .work/run.log 2>&1 &
```

For fish specifically: fish has **no heredocs**, so a `python - <<'PY'` probe
fails with `Expected a string, but found a redirection`, and the abandoned
redirection can swallow the next commands in a reused terminal. Write probe
scripts as files under `.work/` and run them; that is what every `.work/*.py`
file in this repository is for.

### Where the answers land

| Path | Read it for |
|---|---|
| `reports/current_status.html` | the dashboard: metrics, figures, the honest-caveats block |
| `reports/results.json` | every number, keyed; the source of truth for the ledger |
| `figures/localisation.png` | tiled versus ELA Dice, in-domain and per external corpus |
| `figures/cross_dataset.png` | in-domain bars beside zero-shot bars, per corpus |
| `docs/TRANSFER-FINDING.md` | the mechanism behind the chance-level transfer score, and the alternatives that were measured and eliminated |
| `presentation/current_status.md` | talk track and the questions a supervisor will ask |
| `models/forensic_detector.joblib` | the selected model, its threshold, and feature names |

---

## The four models, and who trains them

You train all four; the split is by *contribution*, not by operator.

**1. Classical forensic features + gradient boosting — built and measured.**
This is the command above. Three detectors are fitted (logistic regression,
random forest, histogram gradient boosting) and the winner is chosen on
validation ROC-AUC only. Nothing to do but re-run it.

**2. Fine-tuned CNN backbone — not built yet.**
The plan is a pretrained ResNet-18 with the classifier replaced and only the
last block unfrozen, at 224 px input, batch 8, AMP on. The reason for that
specific shape is the hardware: 3.73 GiB of usable VRAM on an RTX 2050. A
full-resolution CASIA image at batch 8 in fp32 does not fit; a 224 px centre
crop does. Anyone who says otherwise has not run it.

**3. Segmentation U-Net for pixel masks — not built yet.**
This is the honest route to localisation. The current in-domain Dice of 0.131
(tiled) and 0.156 (ELA) comes from re-scoring 128 px tiles with a whole-image
classifier, which can tell that a tile is suspicious but was never asked to draw a
boundary — the tiled hit rate of 0.883 beside a Dice of 0.131 is exactly that
signature. A U-Net trained on the ground-truth masks reports pixel Dice directly,
and CoMoFoD ships 200 masks for exactly that.

**4. Hybrid: CNN deep features + the classical 143 — not built yet.**
Concatenate the backbone's penultimate activation with the forensic vector and
refit the boosting stage. Interesting specifically because of what was measured
today: the forensic groups are dominated by compression statistics that sit 5
sigma away between JPEG and PNG corpora, so the hybrid shows whether deep
features can carry the residual evidence when the compression cues stop
transferring.

For 2-4, the same discipline applies: one command, metrics into
`reports/results.json`, and no number in the report that the ledger cannot
trace.

---

## Checks to run before trusting a result

| Command | Question it answers |
|---|---|
| `.venv/bin/python .work/audit_comofod_loader.py` | is this corpus labelled, grouped and masked the way its publication says? |
| `.venv/bin/python .work/validate_cm_features.py` | is the copy-move group producing non-zero evidence at a survivable cost? |
| `.venv/bin/python .work/probe_inert_features.py` | which of the 143 features are constant on one corpus but alive on the other? |
| `.venv/bin/python .work/measure_cm_contribution.py` | is the repaired copy-move group worth anything, paired on identical rows and an identical split? |
| `.venv/bin/python .work/test_by_tamper_type.py` | does the per-geometry transfer arithmetic still mean what it claims? |
| `.venv/bin/python .work/diagnose_comofod_transfer.py` | is a chance-level transfer score a bug, an unlearnable corpus, or a confound? |

`probe_inert_features.py` is the one worth keeping in the rotation. It found the
dead group: fourteen features constant on CoMoFoD, of which eight — the entire
copy-move group — were constant on *every* corpus, in every run, forever. A
feature that never varies cannot be split on, so it contributes nothing while
still being counted in "143 features". After the repair only six are constant on
CoMoFoD, and three of those are the `global` geometry columns, dead because
CoMoFoD is uniformly 512×512 — a correct measurement rather than a bug.

One more rule that came out of the repair, and it applies to every number in
this project: **compare arms on identical rows.** The ablation figures in
`docs/BASELINE.md` moved when the copy-move group was fixed, but the compression
arm moved too, which the fix cannot have caused. `measure_cm_contribution.py`
answers the question instead by running cm-present and cm-absent on the same
cache, the same split and the same hyper-parameters. Two documentation epochs are
not a controlled experiment.
