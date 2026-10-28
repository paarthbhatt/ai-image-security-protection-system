"""Smoke-test the real-dataset loader against a mock CASIA-style tree."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sources import DatasetUnavailable, load_real_forgery_dataset  # noqa: E402


def build_mock(root: Path, source_dir: Path, count: int = 30) -> None:
    shutil.rmtree(root, ignore_errors=True)
    (root / "Au").mkdir(parents=True)
    (root / "Tp").mkdir(parents=True)
    images = sorted(source_dir.glob("*_authentic.png"))[:count]
    for index, path in enumerate(images):
        shutil.copy(path, root / "Au" / f"Au_ani_{index:03d}.png")
        image = Image.open(path).convert("RGB").resize((64, 64))
        array = np.asarray(image).copy()
        array[10:30, 10:30] = np.roll(array[10:30, 10:30], 12, axis=1)
        Image.fromarray(array).save(root / "Tp" / f"Tp_D_CND_S_{index:03d}.png")


def main() -> None:
    mock = Path("/tmp/mock_casia")
    build_mock(mock, Path(__file__).resolve().parents[1] / "data" / "processed" / "images")

    frame = load_real_forgery_dataset("casia2", mock, mask_root=mock, seed=42)
    print("rows      :", len(frame))
    print("columns   :", list(frame.columns))
    print("labels    :", frame["label"].value_counts().to_dict())
    print("splits    :", frame["split"].value_counts().to_dict())
    print("groups    :", frame["source_id"].nunique())
    crossing = int((frame.groupby("source_id")["split"].nunique() > 1).sum())
    print("leaked    :", crossing)
    print(frame[["sample_id", "source_id", "label", "tamper_type", "split"]].head(4).to_string())

    assert crossing == 0, "source groups leaked across splits"
    assert set(frame["label"].unique()) == {0, 1}, "both classes required"
    assert frame["tamper_type"].isin(["authentic", "unknown_real"]).all()

    missing = Path("/tmp/does_not_exist")
    try:
        load_real_forgery_dataset("casia2", missing)
    except DatasetUnavailable as error:
        print("\nmissing-dataset error OK:", str(error)[:110])
    else:
        raise AssertionError("a missing dataset must raise DatasetUnavailable")

    print("\nAll real-dataset loader checks passed.")


if __name__ == "__main__":
    main()
