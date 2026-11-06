"""Makes ``src/`` importable so the suite can run from the repository root.

Run everything with::

    .venv/bin/python -m unittest discover -s tests -t .

These are not the one-off checks in ``.work/``. A ``.work`` script answers a
question once; a test here states an invariant that a future change must not
break. The distinction matters because the copy-move feature group was dead -
constant, zero-variance, contributing nothing - and every pipeline run still
reported it among "143 features". Nothing in the project failed. That is what
this directory is for.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
