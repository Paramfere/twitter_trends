#!/usr/bin/env python3

"""
Full Trend Intelligence Pipeline
Runs all pipeline stages in the correct order:
1. Fetch trending topics  + rule-based categorisation
2. Generate descriptive reports (AI, velocity, Web3) – executed inside fetch_topics.py
3. (optional) High-engagement tweet scraping & offline filtering

Usage
-----
python scripts/full_pipeline.py           # run lightweight pipeline
python scripts/full_pipeline.py --with-content-analysis  # include tweet scraping

The script is intentionally thin and delegates heavy-lifting to
`scripts/fetch_topics.py`, which already orchestrates the downstream
reports.  This wrapper exists solely to provide a single entry-point so
users don't have to remember multiple commands.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

# Configure logging early so downstream modules inherit the level
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def _build_fetch_command(args: argparse.Namespace) -> List[str]:
    """Return the command list used to invoke ``fetch_topics.py``."""
    cmd: List[str] = [sys.executable, str(Path(__file__).with_name("fetch_topics.py"))]
    if args.with_content_analysis:
        cmd.append("--with-content-analysis")
    return cmd


def main() -> None:
    """Entry-point for the full trend-intelligence pipeline."""
    parser = argparse.ArgumentParser(description="Run the full Twitter-trend pipeline in one go")
    parser.add_argument(
        "--with-content-analysis",
        dest="with_content_analysis",
        action="store_true",
        help="Also scrape & refine high-engagement tweets (slow & paid API calls)",
    )

    cli_args = parser.parse_args()

    fetch_cmd = _build_fetch_command(cli_args)
    LOGGER.info("🚀 Launching fetch_topics.py (%s)", " ".join(fetch_cmd[1:]))

    completed = subprocess.run(fetch_cmd, text=True)

    if completed.returncode != 0:
        LOGGER.error("fetch_topics.py failed with exit-code %s", completed.returncode)
        sys.exit(completed.returncode)

    LOGGER.info("🎉 Full pipeline finished successfully")


if __name__ == "__main__":
    main() 