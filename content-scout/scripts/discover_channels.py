#!/usr/bin/env python3
"""Stub for weekly channel discovery (not needed for v1 launch)."""

from __future__ import annotations

import argparse
import logging

from _common import setup_logging

LOGGER = logging.getLogger("content_scout.discover_channels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="tmp/discovered_channels.json",
        help="Where discovered channels would be written",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # TODO(v1.1): Implement weekly discovery using YouTube API + LLM scoring.
    LOGGER.info("discover_channels is a v1 stub. Output target would be %s", args.output)
    print("discover_channels.py is not implemented for v1 launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
