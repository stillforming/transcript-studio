#!/usr/bin/env python3
"""Stub for channel staleness checks (not needed for v1 launch)."""

from __future__ import annotations

import argparse
import logging

from _common import setup_logging

LOGGER = logging.getLogger("content_scout.check_staleness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels",
        default="config/channels.json",
        help="Channel config path",
    )
    parser.add_argument(
        "--output",
        default="tmp/channel_staleness.json",
        help="Where staleness results would be written",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # TODO(v1.1): Implement staleness monitor against channel upload history.
    LOGGER.info(
        "check_staleness is a v1 stub. Channels=%s output=%s",
        args.channels,
        args.output,
    )
    print("check_staleness.py is not implemented for v1 launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
