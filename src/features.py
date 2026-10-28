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
    Self-matching keypoint structure (SIFT/ORB + ratio test + offset clustering),
    the classical copy-move detector. Unlike the other groups this is structural:
    it counts evidence that a region repeats itself.

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
FEATURE_VERSION = "2024-05-srm-orb-singlescale-bincount"


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

# Keypoints are searched at a fixed working scale. A copy-move pair shares a
# translation, not a scale, so a downscaled image yields the same evidence while
# costing a fraction as much: FAST corner detection on a 768 px photograph takes
# 3.7 s at full resolution versus 0.1 s here.
CM_MAX_SIDE = 256


def _copy_move_working_image(gray_uint8: np.ndarray) -> np.ndarray:
    """Downscale so keypoint search is resolution-independent in cost and behaviour."""
    longest = max(gray_uint8.shape)
    if longest <= CM_MAX_SIDE:
        return gray_uint8

    from skimage.transform import resize

    scale = CM_MAX_SIDE / longest
    height = max(24, round(gray_uint8.shape[0] * scale))
    width = max(24, round(gray_uint8.shape[1] * scale))
    working = resize(gray_uint8, (height, width), preserve_range=True, anti_aliasing=True)
    return working.astype(np.uint8)


def _copymove_features(gray_uint8: np.ndarray) -> list[float]:
    """Self-matching keypoints with offset clustering (classical CMFD evidence).

    Returns zeros when the image is too small or too featureless to support the
    detector, which is the honest answer rather than a fabricated score.
    """
    empty = [0.0] * 8
    if min(gray_uint8.shape) < 24:
        for stat in ("keypoints", "self_matches", "cluster_max", "cluster_count",
                     "cluster_ratio", "offset_entropy", "peak_ratio", "match_density"):
            _register(f"cm_{stat}")
        return empty

    try:
        from skimage.feature import ORB, match_descriptors

        detector = ORB(n_keypoints=256, fast_threshold=0.05, n_scales=1)
        detector.detect_and_extract(
            _copy_move_working_image(gray_uint8).astype(np.float32) / 255.0
        )
        keypoints = detector.keypoints
        descriptors = detector.descriptors
        if descriptors is None or len(descriptors) < 8:
            raise ValueError("too few keypoints")

        matches = match_descriptors(descriptors, descriptors, cross_check=True)
        matches = matches[matches[:, 0] != matches[:, 1]]
        if len(matches) == 0:
            raise ValueError("no self matches")

        offsets = keypoints[matches[:, 1]] - keypoints[matches[:, 0]]
        magnitudes = np.linalg.norm(offsets, axis=1)
        significant = offsets[magnitudes > 2.0]
        if len(significant) == 0:
            raise ValueError("no significant offsets")

        # Two regions are copy-move if many pairs share the same translation.
        buckets = np.rint(significant / 4.0).astype(np.int32)
        _, inverse, counts = np.unique(
            buckets, axis=0, return_inverse=True, return_counts=True
        )
        ordered = np.sort(counts)[::-1]
        cluster_max = int(ordered[0])
        cluster_count = int((counts >= 3).sum())
        probabilities = counts / counts.sum()
        offset_entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum())
        match_density = float(len(significant) / max(1, len(keypoints)))
        del inverse
        values = [
            float(len(keypoints)),
            float(len(significant)),
            float(cluster_max),
            float(cluster_count),
            float(cluster_max / max(1, len(significant))),
            offset_entropy,
            float(ordered[0] / max(1, len(ordered))),
            match_density,
        ]
    except Exception:  # noqa: BLE001 - feature extraction must never abort a run
        values = empty
    for stat in ("keypoints", "self_matches", "cluster_max", "cluster_count",
                 "cluster_ratio", "offset_entropy", "peak_ratio", "match_density"):
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
