#!/usr/bin/env python3
"""Console-script wrappers for FERE pipelines.

After an editable install (``pip install -e .``) the following commands become
available system-wide:

* ``fere-general``   – general pipeline (all categories) with tweet scraping
* ``fere-tech``      – tech-focused pipeline (top-5 tech trends) with tweets
* ``fere-tech-all``  – full tech pipeline (all tech topics) with tweets

The functions below simply forward to the existing scripts so there is no
business-logic duplication.
"""
from __future__ import annotations

import sys
from pathlib import Path
from subprocess import run

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parents[1]  # Repository root


def _exec(cmd: list[str]) -> None:  # noqa: WPS421 (subprocess wrapper)
    """Execute *cmd* and propagate its exit status."""
    run(cmd, check=True)


# ---------------------------------------------------------------------------
# Entry-points
# ---------------------------------------------------------------------------

def general() -> None:
    """Run the *general* pipeline – all categories + tweet scraping."""
    _exec([PYTHON, str(ROOT / "scripts/full_pipeline.py"), "--with-content-analysis"])


def tech() -> None:
    """Run the *tech-focused* pipeline – top-5 tech topics + tweet scraping."""
    _exec([PYTHON, str(ROOT / "scripts/tech_pipeline.py"), "--with-content-analysis"])


def tech_all() -> None:
    """Run the *full tech* pipeline – **all** tech topics + tweet scraping."""
    _exec(
        [
            PYTHON,
            str(ROOT / "scripts/tech_pipeline.py"),
            "--with-content-analysis",
            "--max-topics",
            "1000",
        ],
    ) 