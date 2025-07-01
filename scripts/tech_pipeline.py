#!/usr/bin/env python3
"""Tech-only Twitter Trend Pipeline.

This convenience wrapper runs the two high-level stages targeted at
Technology & Web3 trends only:

1. ``fetch_tech_topics.py`` – scrapes trending topics, applies the
   strict Technology/Web3 filter (+ priority ordering) and stores the
   CSV + report inside a timestamped session directory.
2. *Optional* high-engagement tweet scraping via the content-analysis
   engine – simply forward ``--with-content-analysis`` to the underlying
   fetcher where the heavy lifting already lives.

Example
-------
python scripts/tech_pipeline.py                 # light version
python scripts/tech_pipeline.py --with-content-analysis  # also scrape tweets

The script is intentionally minimal so users only need to remember one
entry-point for the Tech-only flow.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def _build_command(args: argparse.Namespace) -> List[str]:
    """Return the command list used to invoke ``fetch_tech_topics.py``."""
    cmd: List[str] = [sys.executable, str(Path(__file__).with_name("fetch_tech_topics.py"))]
    if args.with_content_analysis:
        cmd.append("--with-content-analysis")
    return cmd


def main() -> None:
    """Run the Tech-only pipeline end to end."""
    parser = argparse.ArgumentParser(description="Run Technology/Web3 trend pipeline")
    parser.add_argument(
        "--with-content-analysis",
        dest="with_content_analysis",
        action="store_true",
        help="Also scrape & refine high-engagement tweets (slow & paid API calls)",
    )

    cli_args = parser.parse_args()

    cmd = _build_command(cli_args)
    LOGGER.info("🚀 Launching fetch_tech_topics.py (%s)", " ".join(cmd[1:]))

    completed = subprocess.run(cmd, text=True)
    if completed.returncode != 0:
        LOGGER.error("fetch_tech_topics.py failed with exit-code %s", completed.returncode)
        sys.exit(completed.returncode)

    LOGGER.info("🎉 Tech-only pipeline finished successfully")


if __name__ == "__main__":
    main() 