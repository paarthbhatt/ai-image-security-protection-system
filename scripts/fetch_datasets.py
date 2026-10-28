"""Check which real forgery benchmarks are present and structurally usable.

Run this after downloading a dataset by hand:

    python scripts/fetch_datasets.py
    python scripts/fetch_datasets.py --extract        # also unpack archives it finds

It never downloads anything on its own: every public benchmark below is either
registration-gated, behind a shared-folder viewer, or too large to fetch at a
usable speed. It tells you exactly what is missing and where it belongs.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sources import DatasetUnavailable, _iter_images, load_real_forgery_dataset  # noqa: E402

ARCHIVE_SUFFIXES = (".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz")

# Archive filenames do not match the dataset names the pipeline expects:
# ``comofod_small.rar`` unpacks to the ``comofod`` dataset, and
# ``casia-20-image-tampering-detection-dataset.zip`` to ``casia2``. Deriving the
# directory from the filename maps them all somewhere the catalogue never looks.
ARCHIVE_TARGETS: tuple[tuple[str, str], ...] = (
    ("casia", "casia2"),
    ("comofod", "comofod"),
    ("coverage", "coverage"),
    ("dso", "dso1"),
)

# name -> (target directory, expected scenes, download instructions)
CATALOGUE: dict[str, tuple[str, int, list[str]]] = {
    "casia2": (
        "Manually obtained — see routes below.",
        400,
        [
            "Target: data/real/casia2/  with subfolders  Au/  and  Tp/",
            "Route A (canonical, registration): http://forensics.idealtest.org/",
            "Route B (Kaggle, needs account):",
            "  pip install kaggle",
            "  kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset",
            "  unzip -q casia-20-image-tampering-detection-dataset.zip -d data/real/",
            "  mv data/real/CASIA2 data/real/casia2",
            "Optional masks: https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth",
            "  -> unzip so that a folder containing 'Groundtruth' sits under data/real/casia2/",
        ],
    ),
    "comofod": (
        "https://www.vcl.fer.hr/comofod/comofod_small.rar  (3.2 GB, direct, no registration)",
        200,
        [
            "Target: data/real/comofod/  (files may sit flat in one folder)",
            "  curl -L -o comofod_small.rar https://www.vcl.fer.hr/comofod/comofod_small.rar",
            "  Extract with: 7z x comofod_small.rar -odata/real/comofod",
            "  (.rar extraction needs p7zip/unrar; it is not installed here yet)",
            "Expected naming: <scene>.png / <scene>_F.png / <scene>_M.png",
            "At ~100 KB/s this download takes roughly 9 hours.",
        ],
    ),
    "coverage": (
        "OneDrive folder — browser download only (returns 403 to scripts).",
        100,
        [
            "Target: data/real/coverage/  with subfolders  image/  and  mask/",
            "Download: https://1drv.ms/f/s!AggVhXcCj1FLhUUyUrqSpV_yI_GH",
            "Open in a browser, download the folder, then unpack it here.",
            "100 pairs, 400x486, TIFF, with ground-truth masks.",
        ],
    ),
}


def _target_for(archive: Path, root: Path) -> Path:
    """Map an archive filename onto the dataset directory it belongs in."""
    name = archive.name.lower()
    for marker, dataset in ARCHIVE_TARGETS:
        if marker in name:
            return root / dataset
    fallback = re.sub(r"[^a-z0-9]+", "_", _strip_suffix(archive.name.lower())).strip("_")
    return root / fallback


def _strip_suffix(name: str) -> str:
    """Remove a full archive suffix, so ``x.tar.gz`` does not become ``x.tar``."""
    for suffix in ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _flatten_single_wrapper(destination: Path) -> str | None:
    """Move ``dest/pkg/Au`` up to ``dest/Au`` when an archive adds one wrapper.

    Kaggle and CASIA releases both nest everything under a single top folder, so
    without this the files land one level deeper than the documented layout and
    the next ``--extract`` run sees a directory it cannot reason about.
    """
    entries = list(destination.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return None
    wrapper = entries[0]
    for child in wrapper.iterdir():
        shutil.move(str(child), str(destination / child.name))
    wrapper.rmdir()
    return wrapper.name


def _unpack(archive: Path, destination: Path) -> bool:
    """Extract an archive using whatever tool is available; True on success."""
    destination.mkdir(parents=True, exist_ok=True)
    suffix = archive.name.lower()
    try:
        if suffix.endswith(".zip") and shutil.which("unzip"):
            subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(destination)], check=True)
            return True
        if suffix.endswith((".7z", ".rar")) and shutil.which("7z"):
            subprocess.run(["7z", "x", str(archive), f"-o{destination}", "-y"], check=True)
            return True
        if shutil.which("bsdtar"):
            subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(destination)], check=True)
            return True
    except subprocess.CalledProcessError:
        return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real forgery datasets are present and usable.")
    parser.add_argument("--root", default="data/real", help="Directory holding the datasets.")
    parser.add_argument("--extract", action="store_true", help="Unpack archives found in the root.")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.extract:
        archives = [p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(ARCHIVE_SUFFIXES)]
        if not archives:
            print(f"No archives found under {root}.")
        for archive in archives:
            target = _target_for(archive, root)
            print(f"Extracting {archive.name} -> {target} ...", flush=True)
            if not _unpack(archive, target):
                print("  FAILED (p7zip or unrar needed for .rar/.7z)")
                continue
            wrapper = _flatten_single_wrapper(target)
            print("  ok" + (f" (unwrapped {wrapper}/)" if wrapper else ""))

    ready, missing = [], []
    for name, (source, expected_scenes, instructions) in CATALOGUE.items():
        directory = root / name
        images = _iter_images(directory) if directory.exists() else []

        print(f"\n=== {name} :: {directory}")
        if not images:
            print("  state: MISSING")
            missing.append(name)
            for line in [f"  source: {source}", *instructions]:
                print(line)
            continue

        try:
            frame = load_real_forgery_dataset(name, directory, mask_root=directory)
        except DatasetUnavailable as error:
            print(f"  state: UNUSABLE ({error})")
            missing.append(name)
            continue

        authentic = int((frame["label"] == 0).sum())
        forged = int((frame["label"] == 1).sum())
        masks = int((frame["mask_path"].astype(str) != "").sum())
        scenes = int(frame["source_id"].nunique())
        leaked = int((frame.groupby("source_id")["split"].nunique() > 1).sum())

        print(f"  state: OK")
        print(f"  images     : {len(frame):,}  (authentic {authentic:,} / forged {forged:,})")
        print(f"  scenes     : {scenes:,}  (expected ~{expected_scenes})")
        print(f"  masks found: {masks:,} of {forged:,} forged")
        print(f"  split leak : {leaked} groups in >1 split")
        if masks == 0:
            print("  note: no masks matched; image-level metrics still work, localisation will not.")
        ready.append(name)

    print("\n" + "=" * 60)
    print(f"Usable: {', '.join(ready) if ready else 'none'}")
    print(f"Missing: {', '.join(missing) if missing else 'none'}")
    if ready:
        print("\nRun:")
        for name in ready:
            print(f"  ./.venv/bin/python -u src/run_pipeline.py --real-dataset {name}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
