"""A small tree that behaves like CASIA v2.0, for tests that must not need 9 GB.

The real archive is the evidence; a fixture is only a shape. It reproduces the
three things the loaders actually key on:

*  ``Au/``, ``Tp/`` and ``Gt/`` folder hints, with ``Gt/<stem>_gt.png`` masks,
   which is exactly how CASIA names them;
*  file names that carry the linkage the real convention carries.
   ``Au_ani_00018`` and ``Tp_D_CND_M_N_ani00018_sec00096_00138`` are the same
   photograph, and ``_casia_sources`` only recognises a name shaped like that,
   so a fixture with invented names would silently test the fallback path;
*  the format mix that the protocol arm exists to deal with: splices arrive as
   TIFF, copy-moves as JPEG, and the unlinked originals arrive as BMP. Because
   ``_linked_scene_keys`` needs 90% of names to parse, every file here follows
   the convention.

Per scene the tree holds three originals and two forgeries, so a build can be
balanced by *dropping* originals - the part of section 4.1.3 that a fixture with
equal classes would never exercise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HEIGHT, WIDTH = 40, 64


def _pattern(index: int) -> np.ndarray:
    """A distinct, non-constant image, so a quality factor can visibly matter."""
    rng = np.random.default_rng(1000 + index)
    base = np.add.outer(np.arange(HEIGHT) * 3, np.arange(WIDTH) * 2) % 251
    noise = rng.integers(0, 24, size=(HEIGHT, WIDTH), dtype=np.uint8).astype(np.int32)
    plane = np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.stack([plane, np.roll(plane, 7, axis=1), np.roll(plane, 3, axis=0)], axis=2)


def _save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        Image.fromarray(array).save(path, format="JPEG", quality=92)
    else:
        Image.fromarray(array).save(path)


def _mask() -> np.ndarray:
    """A rectangular tampered region, shaped the way CASIA shapes them."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[8:24, 16:44] = 255
    return mask


def build_fake_casia(root: Path, scenes: int = 8) -> dict[str, int]:
    """Write a CASIA-shaped corpus under ``root`` and report what it contains."""
    root = Path(root)
    counts = {
        "authentic": 0,
        "tampered": 0,
        "uncompressed_authentic": 0,
        "uncompressed_tampered": 0,
        "masks": 0,
    }

    for index in range(1, scenes + 1):
        # Two originals that the splice below names, so they share a scene, and a
        # third that stands alone and arrives uncompressed: a BMP on the
        # authentic side, which is the case the narrow reading of the protocol
        # would leave as a pure class cue.
        _save(_pattern(index), root / "Au" / f"Au_ani_{index:05d}.jpg")
        _save(_pattern(50 + index), root / "Au" / f"Au_sec_{index:05d}.jpg")
        extra = root / "Au" / f"Au_ani_{index + 100:05d}.bmp"
        _save(_pattern(150 + index), extra)
        counts["authentic"] += 3
        counts["uncompressed_authentic"] += 1

    for index in range(1, scenes + 1):
        # A TIFF splice: uncompressed, and the reason the arm exists.
        splice = f"Tp_S_CND_L_N_ani{index:05d}_sec{index:05d}_{index + 10000:05d}.tif"
        # A JPEG copy-move: already compressed, so it must survive untouched.
        copymove = f"Tp_D_CND_M_N_ani{index:05d}_sec{index:05d}_{index + 20000:05d}.jpg"
        _save(_pattern(300 + index), root / "Tp" / splice)
        _save(_pattern(400 + index), root / "Tp" / copymove)
        counts["tampered"] += 2
        counts["uncompressed_tampered"] += 1

        _save(np.dstack([_mask()] * 3), root / "Gt" / f"{Path(splice).stem}_gt.png")
        counts["masks"] += 1

    return counts
