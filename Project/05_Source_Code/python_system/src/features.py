"""Forensic feature extraction for image-manipulation screening.

The feature bank follows the steganalysis/forensics literature rather than raw
pixel statistics, because global colour moments mostly measure *content* (and are
trivially gamed by a brightness edit). Groups:

``srm``
    High-pass residual responses in the spirit of the Steganalysis Rich Model
    (Fridrich & Kodovsky, IEEE TIFS 2012). Zhou et al. (CVPR 2018) showed these
    residuals are the strong signal for manipulation detection because they
    suppress scene content and expose the noise/quantisation traces that
    splicing and recompression perturb.

``ela``
    Error Level Analysis (Krawetz). Recompress at a fixed quality and describe
    the residual, including block-wise dispersion, which reveals regions with a
    different compression history from their surroundings.

``noise``
    Wavelet-domain noise variance (Donoho's MAD estimator) plus its spatial
    dispersion. A spliced region carries a different sensor/denoising history
    from the host image, so local noise variance becomes bimodal.

``jpeg``
    Blockiness at the 8x8 grid and DCT high-frequency profile, which detect
    recompression and grid misalignment.

``copymove``
    Block-level self-similarity with translation-offset clustering (Webswale &
    Shuller 2007), the classical copy-move detector. Blocks are compared in the
    low-frequency DCT domain rather than bit-exactly, because a re-saved paste is
    similar but not identical. Unlike the other groups this is structural: it
    counts evidence that a region of the image repeats itself elsewhere.

Every group is size-normalised so the vector length never depends on input
resolution.
"""

from __future__ import annotations

import io
import os
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import kurtosis

FEATURE_NAMES: list[str] = []
_NAMES_CACHE: list[str] | None = None

# Bump whenever the values produced by extract_features change for the same
# pixels. Cached matrices carry this tag, so tuning a detector cannot silently
# reuse rows computed by the previous one.
FEATURE_VERSION = "2026-11-dct-block-copymove"


def _register(name: str) -> str:
    """Record a feature name; only the first (name-building) pass is kept."""
    if _NAMES_CACHE is None:
        FEATURE_NAMES.append(name)
    return name


# --- SRM high-pass kernels -------------------------------------------------
# Signed integer kernels; responses are quantised to integers like the original
# SRM so quantisation noise statistics stay meaningful.

_SRM_KERNELS: dict[str, np.ndarray] = {
    "srm_1st_h": np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    "srm_1st_v": np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
    "srm_2nd_h": np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
    "srm_2nd_v": np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),
    "srm_square3": np.array(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
        dtype=np.float32,
    ),
    "srm_square5": np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    "srm_kv": np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ) * 0.25,
    "srm_diag": np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]], dtype=np.float32),
}

_RESIDUAL_STAT_NAMES = [
    "absmean", "std", "kurt", "entropy", "sparsity", "q25", "q50", "q75", "peak",
]

# Residuals are clipped to this range before histogramming, exactly as SRM does.
_CLIP = 8


def _residual_stats(residual: np.ndarray) -> list[float]:
    """Nine scale-free statistics describing one residual channel."""
    quantised = np.clip(np.rint(residual), -_CLIP, _CLIP).astype(np.int32)
    # The residual is already clipped to a 2*_CLIP+1 bin range, so binning is a
    # single linear pass. Empty bins contribute nothing to the entropy or the
    # peak, which makes this identical to sorting and counting unique values.
    counts = np.bincount(quantised.ravel() + _CLIP, minlength=2 * _CLIP + 1)
    probabilities = counts.astype(np.float64) / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum())
    abs_residual = np.abs(residual).astype(np.float64)
    peak = float(probabilities[int(np.argmax(counts))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kurt = float(kurtosis(residual.ravel(), fisher=True, bias=False))
    if not np.isfinite(kurt):
        kurt = 0.0
    q25, q50, q75 = np.percentile(abs_residual, (25, 50, 75))
    return [
        float(abs_residual.mean()),
        float(residual.std()),
        kurt,
        entropy,
        float((abs_residual > 1.0).mean()),
        float(q25),
        float(q50),
        float(q75),
        peak,
    ]


def _srm_features(gray: np.ndarray, image: np.ndarray) -> list[float]:
    values: list[float] = []
    for name, kernel in _SRM_KERNELS.items():
        residual = ndimage.convolve(gray, kernel, mode="nearest")
        for stat_name, value in zip(_RESIDUAL_STAT_NAMES, _residual_stats(residual)):
            values.append(value)
            if len(FEATURE_NAMES) < 4096:
                _register(f"{name}_{stat_name}")
    # Inter-channel residuals expose colour inconsistencies left by splicing,
    # which a per-channel filter cannot see.
    for left, right in ((0, 1), (1, 2), (0, 2)):
        channel_delta = image[..., left].astype(np.float32) - image[..., right].astype(np.float32)
        residual = ndimage.convolve(channel_delta, _SRM_KERNELS["srm_2nd_h"], mode="nearest")
        for stat_name, value in zip(_RESIDUAL_STAT_NAMES, _residual_stats(residual)):
            values.append(value)
            if len(FEATURE_NAMES) < 4096:
                _register(f"chroma{left}{right}_{stat_name}")
    return values


# --- Error Level Analysis --------------------------------------------------


def _recompress_error(gray_uint8: np.ndarray, quality: int) -> np.ndarray:
    image = Image.fromarray(gray_uint8, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer), dtype=np.float32)
    return np.abs(gray_uint8.astype(np.float32) - recompressed)


def _block_stats(array: np.ndarray, block: int = 8) -> tuple[float, float, float]:
    """Mean, spread and max-to-mean ratio of block-wise means of ``array``."""
    h, w = array.shape
    blocks = [
        array[y : y + block, x : x + block].mean()
        for y in range(0, max(1, h - block + 1), block)
        for x in range(0, max(1, w - block + 1), block)
        if array[y : y + block, x : x + block].size
    ]
    if not blocks:
        return 0.0, 0.0, 0.0
    means = np.asarray(blocks, dtype=np.float64)
    overall = float(means.mean())
    return overall, float(means.std()), float(means.max() / (overall + 1e-6))


def _ela_features(gray_uint8: np.ndarray) -> list[float]:
    values: list[float] = []
    for quality in (90, 75):
        error = _recompress_error(gray_uint8, quality)
        mean, spread, ratio = _block_stats(error)
        values.extend(
            [
                float(error.mean()),
                float(error.std()),
                float(np.percentile(error, 95)),
                float(error.max()),
                mean,
                spread,
                ratio,
                float(kurtosis(error.ravel(), fisher=True, bias=False) or 0.0),
            ]
        )
        for stat in ("mean", "std", "p95", "max", "blockmean", "blockspread", "max_ratio", "kurt"):
            _register(f"ela_q{quality}_{stat}")
    return values


# --- Noise inconsistency ---------------------------------------------------


def _wavelet_noise_variance(gray: np.ndarray) -> list[float]:
    """Noise sigma from the HH sub-band at two scales (Donoho & Johnstone)."""
    sigmas: list[float] = []
    current = gray.astype(np.float32)
    for _ in range(2):
        if current.shape[0] < 4 or current.shape[1] < 4:
            sigmas.append(0.0)
            continue
        h, w = current.shape[0] // 2 * 2, current.shape[1] // 2 * 2
        block = current[:h, :w]
        hh = (
            block[0::2, 0::2] - block[0::2, 1::2] - block[1::2, 0::2] + block[1::2, 1::2]
        ) / 2.0
        sigmas.append(float(np.median(np.abs(hh)) / 0.6745))
        current = ndimage.uniform_filter(block, size=2)[::2, ::2]
    return sigmas


def _noise_features(gray_uint8: np.ndarray) -> list[float]:
    gray = gray_uint8.astype(np.float32)
    sigmas = _wavelet_noise_variance(gray)
    # Dispersion of block-wise noise estimates: the tell-tale of a spliced region.
    block_sigmas: list[float] = []
    h, w = gray.shape
    step = max(8, min(h, w) // 4)
    for y in range(0, max(1, h - 8 + 1), step):
        for x in range(0, max(1, w - 8 + 1), step):
            block = gray[y : y + step, x : x + step]
            if block.shape[0] >= 4 and block.shape[1] >= 4:
                block_sigmas.append(_wavelet_noise_variance(block)[0])
    dispersion = float(np.std(block_sigmas)) if len(block_sigmas) > 1 else 0.0
    spread = float(np.ptp(block_sigmas)) if block_sigmas else 0.0
    values = [*sigmas, dispersion, spread, float(len(block_sigmas))]
    for stat in ("sigma_scale1", "sigma_scale2", "block_dispersion", "block_range", "block_count"):
        _register(f"noise_{stat}")
    return values


# --- JPEG grid artefacts ---------------------------------------------------


def _blockiness(gray: np.ndarray) -> float:
    """Ratio of 8x8-grid-boundary gradient energy to interior gradient energy."""
    if gray.shape[0] < 17 or gray.shape[1] < 17:
        return 0.0
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    h_boundary = horizontal[:, 7::8].mean()
    h_interior = np.delete(horizontal, np.s_[7::8], axis=1).mean()
    v_boundary = vertical[7::8, :].mean()
    v_interior = np.delete(vertical, np.s_[7::8], axis=0).mean()
    ratio = (h_boundary / (h_interior + 1e-6) + v_boundary / (v_interior + 1e-6)) / 2.0
    return float(ratio)


def _jpeg_features(gray: np.ndarray) -> list[float]:
    from scipy.fftpack import dct

    coefficients = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
    abs_coefficients = np.abs(coefficients)
    hf = float(abs_coefficients[4:, 4:].mean())
    total = float(abs_coefficients.mean()) + 1e-8
    # Fraction of near-zero AC coefficients: higher after aggressive quantisation.
    ac = abs_coefficients.copy()
    ac[:1, :1] = 0.0
    zero_ratio = float((ac < 0.5).mean())
    values = [hf, hf / total, zero_ratio, _blockiness(gray)]
    for stat in ("hf_energy", "hf_ratio", "ac_zero_ratio", "blockiness"):
        _register(f"jpeg_{stat}")
    return values


# --- Copy-move structural evidence ----------------------------------------
#
# This group used to self-match ORB keypoints, and it never once worked. The code
# asked ``match_descriptors(descriptors, descriptors, cross_check=True)``, whose
# best match for keypoint *i* is always keypoint *i* itself at distance zero; the
# next line filtered the identity pairs out, so every image raised "no self
# matches", and the surrounding ``except Exception`` turned that into eight zeros.
# The group was therefore silent on every corpus and in every committed result,
# while still looking like a legitimate "no evidence here" answer.
#
# What replaces it is the classical block matcher (Webswale & Shuller, 2007),
# chosen on measurement rather than preference. On 50 CASIA copy-move forgeries
# against 50 authentic images (.work/sweep_copymove_floor.py): cm_offset_entropy
# reaches AUC 0.749 and cm_matched_pairs 0.722, where the repaired ORB path scored
# 0.48-0.55 (chance) and bit-exact block equality found *zero* matches. Exact
# equality fails for a reason worth knowing: a paste that does not land on the
# JPEG 8x8 grid is requantised on re-save, so copied pixels are similar but never
# identical. Low-frequency DCT matching survives that.
#
# The limit of the method belongs in the report, not a footnote: offset clustering
# detects *translated* copies. CoMoFoD's own readme.txt states that the copied
# region is transformed in 160 of its 200 scenes (ROTATION, SCALING, DISTORTION,
# COMBINATION), so this group is structurally blind to four fifths of that corpus.

CM_BLOCK = 16             # block edge, in pixels
CM_STRIDE = 8             # overlapping, so a small paste still yields several blocks
CM_TEXTURE_FLOOR = 25.0   # grey-level std; flatter blocks match anything (measured)
CM_SIMILARITY = 0.98      # cosine similarity of unit-normed low-frequency DCT vectors
CM_MIN_SEPARATION = 4     # in blocks: a copy must be displaced, not neighbouring texture
CM_OFFSET_BUCKET = 1.0    # translation clustering resolution, in blocks
CM_CANDIDATE_WINDOW = 96  # neighbours compared per descriptor after the projection sort
CM_MAX_PAIRS = 200_000    # past this the matches are repetitive texture, not a forgery

CM_STATISTICS = (
    "blocks",
    "matched_pairs",
    "pair_ratio",
    "cluster_max",
    "cluster_groups",
    "cluster_ratio",
    "offset_entropy",
    "peak_ratio",
)


def _cm_blocks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Overlapping texture-bearing blocks and the grid coordinates they came from."""
    rows = (gray.shape[0] - CM_BLOCK) // CM_STRIDE + 1
    cols = (gray.shape[1] - CM_BLOCK) // CM_STRIDE + 1
    if rows < 3 or cols < 3:
        return None

    strides = (gray.strides[0] * CM_STRIDE, gray.strides[1] * CM_STRIDE, *gray.strides)
    windows = np.lib.stride_tricks.as_strided(
        gray, shape=(rows, cols, CM_BLOCK, CM_BLOCK), strides=strides
    )
    blocks = np.asarray(windows, dtype=np.float64).reshape(rows * cols, CM_BLOCK * CM_BLOCK)
    coordinates = np.stack(
        (np.repeat(np.arange(rows), cols), np.tile(np.arange(cols), rows)), axis=1
    )

    keep = blocks.std(axis=1) >= CM_TEXTURE_FLOOR
    if int(keep.sum()) < 8:
        return None
    return blocks[keep], coordinates[keep]


def _cm_descriptors(blocks: np.ndarray) -> np.ndarray:
    """Unit-normed low-frequency DCT coefficients, DC term dropped.

    Dropping DC is what lets a bright paste match a dark host: the remaining
    fifteen coefficients describe the patch's structure, not its level.
    """
    from scipy.fft import dctn

    coefficients = dctn(
        blocks.reshape(-1, CM_BLOCK, CM_BLOCK), type=2, norm="ortho", axes=(1, 2)
    )
    low = coefficients[:, :4, :4].reshape(len(blocks), 16)[:, 1:]
    norms = np.linalg.norm(low, axis=1, keepdims=True)
    return low / np.where(norms == 0.0, 1.0, norms)


def _cm_candidate_pairs(descriptors: np.ndarray) -> np.ndarray:
    """Near-duplicate block pairs, blocked by a projection sort instead of O(n^2).

    Similar descriptors land close together once sorted by any fixed direction, so
    comparing each block with its next ``CM_CANDIDATE_WINDOW`` neighbours finds the
    pairs that matter at a fraction of the all-pairs cost -- which is the
    difference between a feature that can run over 12,614 images and one that
    cannot.
    """
    direction = descriptors.mean(axis=0)
    direction /= max(1e-9, float(np.linalg.norm(direction)))
    order = np.argsort(descriptors @ direction)
    ordered = descriptors[order]

    collected: list[np.ndarray] = []
    total = 0
    for offset in range(1, CM_CANDIDATE_WINDOW):
        scores = (ordered[:-offset] * ordered[offset:]).sum(axis=1)
        hits = np.flatnonzero(scores >= CM_SIMILARITY)
        if hits.size == 0:
            continue
        collected.append(np.stack([order[hits], order[hits + offset]], axis=1))
        total += int(hits.size)
        if total > CM_MAX_PAIRS:
            break
    if not collected:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(collected, axis=0), axis=0)


def _copymove_features(gray_uint8: np.ndarray) -> list[float]:
    """Translation-consistent duplicate blocks: the classical copy-move signature.

    Zeros mean "no matching blocks survived the separation filter", which is a
    measurement. There is deliberately no blanket ``except`` here, because that
    is exactly the shape of the defect this function replaces.
    """
    found = _cm_blocks(gray_uint8.astype(np.float64))
    if found is None:
        return [0.0] * len(CM_STATISTICS)
    blocks, coordinates = found

    values = [float(len(blocks))] + [0.0] * (len(CM_STATISTICS) - 1)
    pairs = _cm_candidate_pairs(_cm_descriptors(blocks))
    if len(pairs) > 0:
        offsets = coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]]
        displaced = (np.abs(offsets[:, 0]) + np.abs(offsets[:, 1])) >= CM_MIN_SEPARATION
        pairs, offsets = pairs[displaced], offsets[displaced]
        if len(pairs) > 0:
            buckets = np.rint(offsets / CM_OFFSET_BUCKET).astype(np.int32)
            _, counts = np.unique(buckets, axis=0, return_counts=True)
            ordered_counts = np.sort(counts)[::-1]
            probabilities = counts / counts.sum()
            values[1] = float(len(pairs))
            values[2] = float(len(pairs) / len(blocks))
            values[3] = float(ordered_counts[0])
            values[4] = float((counts >= 3).sum())
            values[5] = float(ordered_counts[0] / len(pairs))
            values[6] = float(-(probabilities * np.log2(probabilities + 1e-12)).sum())
            values[7] = float(ordered_counts[0] / len(counts))

    for stat in CM_STATISTICS:
        _register(f"cm_{stat}")
    return values


# --- Global context --------------------------------------------------------


def _global_features(image: np.ndarray, gray: np.ndarray) -> list[float]:
    normalised = image.astype(np.float32) / 255.0
    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    laplacian = ndimage.laplace(gray)
    values = [
        float(gray.mean()), float(gray.std()),
        float(gradient.mean()), float(gradient.std()),
        float((gradient > 0.18).mean()),
        float(laplacian.var()),
        float(normalised[..., 0].mean() - normalised[..., 2].mean()),
        float(np.std(normalised[..., 0] - normalised[..., 2])),
        float(image.shape[0]),
        float(image.shape[1]),
        float(image.shape[0] / max(1, image.shape[1])),
    ]
    for stat in ("gray_mean", "gray_std", "grad_mean", "grad_std", "edge_density",
                 "laplacian_var", "chroma_mean", "chroma_std", "height", "width", "aspect"):
        _register(f"global_{stat}")
    return values


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return one fixed-length forensic feature vector for an RGB image."""
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.uint8)
    gray_uint8 = np.asarray(Image.fromarray(image).convert("L"), dtype=np.uint8)
    gray = gray_uint8.astype(np.float32) / 255.0

    values: list[float] = []
    values.extend(_srm_features(gray, image.astype(np.float32) / 255.0))
    values.extend(_ela_features(gray_uint8))
    values.extend(_noise_features(gray_uint8))
    values.extend(_jpeg_features(gray))
    values.extend(_copymove_features(gray_uint8))
    values.extend(_global_features(image, gray))
    vector = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return vector


def feature_names() -> list[str]:
    """Feature-name list matching :func:`extract_features`, built once."""
    global _NAMES_CACHE
    if _NAMES_CACHE is None:
        FEATURE_NAMES.clear()
        probe = np.zeros((64, 64, 3), dtype=np.uint8)
        probe[::2, ::2] = 200
        probe[1::2, 1::2] = 60
        extract_features(probe)
        _NAMES_CACHE = list(FEATURE_NAMES)
        FEATURE_NAMES.clear()
        FEATURE_NAMES.extend(_NAMES_CACHE)
    return list(_NAMES_CACHE)


def _extract_one(path: str, transform=None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if transform is not None:
        image = transform(image)
    return extract_features(np.asarray(image, dtype=np.uint8))


def _worker_count(count: int) -> int:
    """Leave a core free, and skip the process pool when it cannot pay for itself."""
    if count < 32:
        return 1
    return max(1, min(12, (os.cpu_count() or 1) - 1))


def extract_paths(paths, transform=None, stage: str = "extracted") -> np.ndarray:
    """Build one feature row per image path, spreading CPU-bound work across cores.

    Called serially this is the slowest part of the pipeline by an order of
    magnitude, so extraction is the one place worth a process pool.
    """
    paths = [str(path) for path in paths]
    if not paths:
        return np.empty((0, len(feature_names())), dtype=np.float32)

    workers = _worker_count(len(paths))
    if workers == 1:
        return np.vstack([_extract_one(path, transform) for path in paths])

    from joblib import Parallel, delayed

    chunk = max(64, workers * 8)
    rows: list[np.ndarray] = []
    for start in range(0, len(paths), chunk):
        block = paths[start : start + chunk]
        rows.extend(
            Parallel(n_jobs=workers, prefer="processes")(
                delayed(_extract_one)(path, transform) for path in block
            )
        )
        print(f"  {stage} {len(rows)}/{len(paths)} images", flush=True)
    return np.vstack(rows)


def extract_frame(frame) -> tuple[np.ndarray, list[str]]:
    """Extract feature vectors for every ``image_path`` in ``frame``."""
    names = feature_names()
    matrix = extract_paths(frame["image_path"])
    if matrix.shape[1] != len(names):
        raise AssertionError(
            f"Feature count mismatch: extracted {matrix.shape[1]} values but "
            f"{len(names)} names were registered."
        )
    return matrix, names
