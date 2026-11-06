# Zero-shot transfer to CoMoFoD is at chance — and here is the mechanism

Measured 2026-11-06 with `FEATURE_VERSION = 2026-11-dct-block-copymove`, one
command, no hand-adjusted number. Every figure below is a key in
`reports/results.json` or a line printed by one of the four scripts named at the
end.

This document exists because the honest result of this experiment is a negative,
and a negative reported without its mechanism is indistinguishable from a broken
pipeline.

## What was run

CASIA v2.0 trained the detector. CoMoFoD small v2 was scored once and never
touched anything:

- the model is the one fitted on CASIA (`models/forensic_detector.joblib`);
- the decision threshold 0.36 was chosen on CASIA's validation split;
- the region thresholds (tiled 0.25, ELA 0.75) were likewise chosen on CASIA;
- every CoMoFoD row is `split == "test"`, and at most 4 of its ~51
  post-processing variants per scene are scored, so 200 photographs are reported
  as 200 scenes rather than as a 10,200-sample benchmark.

Protocol code: `src/cross_dataset.py`. Its docstring is the contract.

## The two numbers side by side

| | In-domain CASIA test | Zero-shot CoMoFoD |
|---|---:|---:|
| images / independent scenes | 1,935 | 1,600 / 200 |
| balanced accuracy | 0.8765 | 0.4988 |
| ROC-AUC | **0.9430** (95% CI 0.933–0.953) | **0.5062** |
| MCC | 0.7399 | −0.0029 |
| precision at the fixed threshold | 0.7976 | 0.4992 |
| recall | 0.8975 | 0.7475 |
| localisation Dice (tiled) | 0.1313 | 0.051 |
| localisation hit rate (tiled) | 0.883 | 0.75 |
| falsely flagged area on authentic images | 0.078 | 0.353 |

AUC 0.506 with MCC −0.003 is chance. The recall of 0.7475 is not skill: at the
inherited threshold the model flags 75% of CoMoFoD forgeries *and* 75% of
CoMoFoD authentic images, which is why precision is 0.4992 on a 50/50 set and
why the falsely flagged area tripled. Reading recall without that pairing would
have turned a failure into a victory lap.

## Four explanations, tested in order

### 1. Is the corpus mislabelled? No.

`.work/audit_comofod_loader.py` re-derives the composition independently of
`src/sources.py` and passes: 10,200 labelled rows, 5,200 authentic / 5,000
forged, 200 scenes, all 5,000 forgeries masked, and every forged row's
`tamper_type` agreeing with the scene's section in the corpus's own `readme.txt`
(translation 40, rotation 40, scaling 40, distortion 39, combination 41).

### 2. Is CoMoFoD simply unlearnable with these features? No.

Training on CoMoFoD itself and testing on 46 held-out scenes gives **ROC-AUC
0.709**, balanced accuracy 0.620 — same 143 features, same model class. So the
corpus contains forgery evidence these features can represent. The failure is
specific to *transfer*, not to the data.

### 3. Is it a geometry mismatch? No, and this is the hypothesis we killed.

CoMoFoD's `readme.txt` documents five transformation families. A translation
offset cue can only address one of them, so the attractive story was "the
detector works on the translation subset and is blind to the rest". Measured per
family, forgeries against that corpus's authentic images:

| Tamper geometry | forged n | transfer ROC-AUC | tiled Dice |
|---|---:|---:|---:|
| distortion | 156 | 0.553 | 0.079 |
| scaling | 160 | 0.528 | 0.038 |
| translation | 160 | 0.500 | 0.042 |
| combination | 164 | 0.486 | 0.068 |
| rotation | 160 | 0.466 | 0.029 |

All five are at chance, and the *best* one is distortion — the geometry a
translation matcher is least able to describe. The image-level version of the
story is therefore false; we would have published it if we had not measured it.
The localisation column keeps a weaker version alive: rotation and scaling are
the two worst families to localise (Dice 0.029 and 0.038 against 0.079 for
distortion), which is consistent with geometric transforms displacing evidence
outside the tile grid a whole-image classifier reasons about.

### 4. Is it a post-processing confound? Yes.

CoMoFoD ships each scene in ~24 post-processing states, applied to both the
original and the forgery. Stratifying the same transfer scores by that nuisance
variable separates the corpus cleanly:

| Subset | n | ROC-AUC | mean score forged vs authentic |
|---|---:|---:|---|
| `none` — scene never reprocessed | 82 | **0.825** | 0.699 vs 0.293 |
| 24 degraded codes (BC, CA, CR, IB, JC, NA ×1–4) | 1,518 | 0.415–0.593 | overlapping |

Variance in the score explained by the post-processing code alone: **0.193**.
Variance explained by the forgery label alone: **0.000**.

The detector is not failing to see forgeries. It is reading compression history,
and on CoMoFoD compression history is assigned by the code column rather than by
the class. Where that confound is absent (`none`), the same frozen model reaches
0.825 without having seen a single CoMoFoD image.

"Yes" here answers *what the score is reading on CoMoFoD*. It is not a claim that
removing the confound from the training side repairs transfer. That intervention
was run, and it is section 5.

### 5. Does normalising the training corpus fix it? No — and this was run, not argued

The obvious remedy for section 4 is to stop CASIA encoding compression history at
all. `src/modified_casia.py` does exactly that, following the base paper's
protocol (Yadav & Vishwakarma 2024, §4.1.3): every uncompressed CASIA image —
3,059 tampered TIFFs and 44 of the 54 authentic BMPs — is re-encoded to JPEG at a
per-image quality factor drawn at random from 50–100, and the authentic class is
thinned to 5,123 for balance. 3,103 conversions, 10,246 rows, manifest
`ff871121a83bcc7d…` recorded in `reports/modified_casia_build.json`.

Because the format cue was a *class* cue before (60% of tampered images were
TIFF, and TIFF was 76% splicing), the conversion covers both classes. Converting
only the tampered side, which is the literal reading of one sentence in the paper,
would leave "is BMP" as a pure authentic-only signal — the arm exists to remove
cues, not to relocate them. `--tampered-only` reproduces the narrow reading if a
reviewer wants it.

The whole pipeline then re-ran on that corpus, same frozen protocol, same
test-only CoMoFoD scoring:

| | CASIA as distributed | modified CASIA (parity arm) |
|---|---:|---:|
| selected model | hist_gradient_boosting | random_forest |
| in-domain ROC-AUC | 0.9430 | 0.9211 |
| in-domain MCC | 0.7399 | 0.7260 |
| zero-shot CoMoFoD ROC-AUC | 0.5062 | **0.5141** |
| zero-shot MCC | −0.0029 | −0.0362 |
| zero-shot per-geometry range | 0.466–0.553 | 0.492–0.531 |

So the parity corpus scores 0.022 lower in-domain, on a different number of rows
and a different set of held-out scenes — the thinning to 5,123 originals is part of
the protocol, so this is not a controlled A/B and must not be quoted as the price
of the format cue. The controlled price is the 0.698 forensic-only arm in
`docs/BASELINE.md`. What matters here is the second column: transfer did not move.
0.5141 is as much chance as 0.5062, and that closes the last cheap explanation
this project could have tested by itself. The confound of
section 4 is real but it lives on the *evaluation* side — CoMoFoD assigns
compression history independently of class, and no amount of tidying the training
corpus makes a compression-reading score prefer forgery to authentic there. It is
also worth stating what this experiment is *not*: it is not a fairer CASIA
baseline for comparison with the base paper's Table 9 (that is its other purpose,
and `docs/BASELINE.md` uses it that way), and it does not touch the leakage rules —
the group check still passes, because `source_id` is carried through the
conversion rather than re-derived from the new file path.

## Why the features carry compression history in the first place

Mean absolute z-shift of each feature group between corpora, z on the CASIA
scale — that is, how far each group moves simply by changing corpus:

| Group | mean \|z\| shift |
|---|---:|
| chroma12 | 5.45 |
| chroma01 | 5.19 |
| chroma02 | 3.72 |
| noise | 1.72 |
| cm | 0.77 |
| ela | 0.68 |
| global | 0.59 |
| srm | 0.30 |
| jpeg | 0.20 |

The three chroma groups are the most discriminative features the model has
(`chroma01_kurt` is the top importer at 0.067) and they move 5σ between a JPEG
corpus and a PNG corpus. CASIA's in-domain advantage and CoMoFoD's chance-level
transfer have the same origin: the detector learned the statistics of a
distribution, and the distribution, not the forgery, is what changed.

A control inside `.work/normalize_probe.py` sharpens that claim. Re-encoding both
CASIA classes losslessly to PNG leaves the score identical to four decimal places
(0.9565 either way on its subsample), while re-encoding both through one JPEG
generation at q90 drops it to 0.8729. Nothing in the feature vector can read a
file extension — only what decoding left in the pixels — so the shortcut this
project has been calling "file format" is really **compression history**, which
is exactly the variable CoMoFoD's 51-variant design assigns independently of
class. The two findings are the same finding.

## The copy-move defect found on the way, and what it cost

`cm_*` was reported as zero on CoMoFoD in the first diagnosis (mean \|z\| shift
0.00). Investigating that led to a defect in `src/features.py`: ORB descriptor
self-matching with `cross_check=True` returns the identity mapping, the next line
filtered identity matches out, and the resulting "no self matches" error was
swallowed by a blanket `except Exception`, so **all eight copy-move features were
identically zero for every image in every run in the project's history**. They
were still being counted in "143 features", and a feature that never varies
cannot be split on.

Replaced by a block matcher over unit-normed low-frequency DCT descriptors with
offset clustering. The replacement was chosen between two measured alternatives:
a fixed ORB matcher stayed near chance (0.48–0.64) and bit-exact block matching
found zero matches on both corpora, because a pasted block that is not aligned to
the 8×8 JPEG grid is requantised and never bit-exact again.

Paired on identical rows, identical split, identical hyper-parameters
(`.work/measure_cm_contribution.py`):

| Arm | features | ROC-AUC |
|---|---:|---:|
| full set, cm repaired | 143 | 0.9430 |
| full set, cm deleted | 135 | 0.9472 |
| `srm`+`noise`+`cm` | 85 | 0.6979 |
| `srm`+`noise` only | 77 | 0.6608 |

The repaired group is worth **+0.037 AUC** where it is the only company it keeps,
and **−0.004 AUC** in the full set, where the compression-flavoured features
already dominate and eight more columns cost the boosting model a little
dilution. In production only 2 of the 8 columns reach the top-40 importance list
(`cm_blocks` at rank 22 and `cm_peak_ratio` at rank 29); the strongest single
statistic by itself is `cm_blocks` at univariate ROC-AUC 0.655, followed by
`offset_entropy` 0.613 and `matched_pairs` 0.611.

The group stays in. Deleting it would buy 0.004 AUC by removing the one evidence
channel the project's own base paper treats as the interesting problem, and
reporting a headline that is higher because a feature was quietly dropped is the
move this document exists to forbid.

## What would actually improve transfer

Stated as work, not as claims:

1. **Compression-history-normalised training.** Already done, and it did not
   work: section 5 retrained on CASIA with every uncompressed image re-encoded at
   a random quality factor. In-domain 0.943 → 0.921, zero-shot 0.506 → 0.514.
   The earlier format-normalised q90 experiment (0.943 → 0.873) measured the price
   of the shortcut; neither experiment moved transfer, so this line is kept as a
   closed option, not a proposal. What is left of the idea is the *evaluation*
   half: normalising the incoming image to one codec state before scoring, which
   is a different intervention and has not been tested here.
2. **Domain-adversarial or per-group normalisation** of the feature vector at
   inference, so a 5σ chroma shift is not read as evidence.
3. **A real segmentation model** (U-Net on the 200 CoMoFoD masks). Tile
   rescoring with a whole-image classifier reports Dice 0.13 in-domain and 0.051
   on CoMoFoD; it was never asked to draw a boundary. Not built — see
   `docs/TEAM-WORKFLOW.md` for what exists and what does not.
4. **More external corpora.** Columbia and COVERAGE are still blocked on a manual
   browser download with a licence form and a shared password
   (`docs/REAL-DATASETS.md`). One external corpus is an anecdote; the base paper
   we compare against reports one corpus too, which is the actual gap in the
   literature rather than a flaw of ours.

What we will not do: fine-tune on CoMoFoD and report the result as transfer. The
0.709 within-CoMoFoD figure is labelled as its own experiment for exactly that
reason.

## Reproducing this

```bash
# one command rebuilds every number in the tables above except section 4's
# stratification and section 5's arm, and takes ~12 minutes when the feature
# caches are warm
.venv/bin/python -u src/run_pipeline.py \
    --real-dataset casia2 \
    --localise-limit 240 \
    --cross-dataset comofod \
    --cross-localise-limit 120 \
    --cross-variants 4

# section 5: build the parity corpus, then run the identical protocol on it
.venv/bin/python src/modified_casia.py build          # one decode + one JPEG encode per uncompressed image
.venv/bin/python src/modified_casia.py verify         # re-checks dimensions and the manifest digest
.venv/bin/python -u src/run_pipeline.py \
    --real-dataset casia2_mod --arm casia2_mod \
    --cross-dataset comofod \
    --localise-limit 240 --cross-localise-limit 120 --cross-variants 4
# --arm moves every artifact root to reports/arms/casia2_mod, figures/arms/... ,
# so a side experiment cannot overwrite the evidence the headline cites

.venv/bin/python .work/audit_comofod_loader.py          # explanation 1
.venv/bin/python .work/diagnose_comofod_transfer.py     # explanations 2, 4, and the z-shifts
.venv/bin/python .work/measure_cm_contribution.py       # the paired cm on/off comparison
.venv/bin/python .work/validate_cm_features.py          # proof the cm group is alive
.venv/bin/python -m unittest discover -s tests -t .     # the durable suite, incl. the arm's rules
```

Result keys: `test`, `localisation`, `cross_dataset.datasets.comofod.image_level`,
`cross_dataset.datasets.comofod.by_tamper_type`,
`cross_dataset.datasets.comofod.localisation.tiled.by_family`.

## Limits of this finding

- Effective sample size is 200 scenes, not 1,600 images.
- Single seed (42). No multi-seed spread is reported.
- Six features are still constant on CoMoFoD (`srm_1st_{h,v}_sparsity`,
  `noise_block_count`, and the three `global` geometry columns, which are dead
  because the corpus is uniformly 512×512). Fourteen were constant before the cm
  repair.
- The `none` subset is 82 rows (26 forged). Its 0.825 is a real signal and a
  small sample at the same time; the correct claim is "the confound is visible",
  not "transfer is solved when compression is controlled".
