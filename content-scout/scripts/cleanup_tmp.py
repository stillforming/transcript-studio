#!/usr/bin/env python3
"""Clean up temporary Content Scout pipeline artifacts."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from _common import resolve_path, setup_logging

LOGGER = logging.getLogger("content_scout.cleanup_tmp")

REMOVE_DIRS = ["downloads", "frames", "transcripts"]
REMOVE_FILES = [
    "video_list.json",
    "windowed_frames.json",
    "annotations.json",
    "_pipeline_state.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp-dir", default="tmp", help="Temporary directory root")
    parser.add_argument("--keep-state", action="store_true", help="Keep _pipeline_state.json")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    tmp_dir = resolve_path(args.tmp_dir)
    removed = 0

    for directory in REMOVE_DIRS:
        target = tmp_dir / directory
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed += 1
            LOGGER.info("Removed %s", target)

    for filename in REMOVE_FILES:
        if args.keep_state and filename == "_pipeline_state.json":
            continue
        target = tmp_dir / filename
        if target.exists():
            target.unlink(missing_ok=True)
            removed += 1
            LOGGER.info("Removed %s", target)

    print(f"Cleanup complete, removed {removed} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
