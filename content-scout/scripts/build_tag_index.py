#!/usr/bin/env python3
"""Build or update reverse tag index for stored frame assets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from _common import ensure_dir, load_json, resolve_path, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.build_tag_index")

# Categories are classification labels, not content tags — exclude from tag index
CATEGORY_NAMES = {"CHART_VISUAL", "TALKING_HEAD", "GRAPHIC", "SCREEN", "TABLE", "SLIDE", "FILLER", "CHART"}

# Known ticker symbols — kept uppercase to distinguish from words
KNOWN_TICKERS = {
    "SPY", "SPX", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "META", "AMZN", "MSFT",
    "GOOG", "GOOGL", "NFLX", "HD", "WMT", "GLD", "TLT", "VIX", "OVX", "DXY",
    "BTC", "ETH", "PLTR", "ORCL", "OXY", "IGV", "FVRR", "IWM", "/CL", "/ES",
    "/NQ", "/GC", "/SI", "XLF", "XLE", "XLK", "XLV", "COIN", "MSTR", "SMCI",
    "ARM", "AVGO", "MU", "INTC", "COST", "CRM", "UBER", "BA", "JPM", "GS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Path to daily _index.json")
    parser.add_argument(
        "--tag-index",
        default="content-vault/tags/tag-index.json",
        help="Path to global tag index JSON",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def normalize_tag(value: Any) -> str:
    """Normalize a tag: tickers stay uppercase, everything else lowercased."""
    tag = str(value or "").strip()
    if not tag:
        return ""
    # Skip category names entirely
    if tag.upper() in CATEGORY_NAMES:
        return ""
    # Keep known tickers uppercase
    if tag.upper() in KNOWN_TICKERS:
        return tag.upper()
    # Lowercase everything else for dedup
    return tag.lower()


def collect_tags(item: dict[str, Any]) -> set[str]:
    tags: set[str] = set()

    raw_tags = item.get("tags", [])
    if isinstance(raw_tags, list):
        for raw in raw_tags:
            tag = normalize_tag(raw)
            if tag:
                tags.add(tag)

    ticker = normalize_tag(item.get("ticker"))
    if ticker:
        tags.add(ticker)

    # Deliberately NOT adding category — it's a classification label, not a content tag

    return tags


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    index_path = resolve_path(args.index)
    tag_index_path = resolve_path(args.tag_index)

    daily_entries = load_json(index_path, default=[])
    if not isinstance(daily_entries, list):
        raise ValueError(f"Index must be a list: {index_path}")

    existing = load_json(tag_index_path, default={})
    if not isinstance(existing, dict):
        existing = {}

    updates = 0

    for item in daily_entries:
        if not isinstance(item, dict):
            continue

        path_value = str(item.get("path") or "").strip()
        if not path_value:
            continue

        for tag in collect_tags(item):
            bucket = existing.setdefault(tag, [])
            if path_value not in bucket:
                bucket.append(path_value)
                updates += 1

    # Clean up: remove stale category entries from prior runs + deduplicate paths
    cleaned = 0
    for tag in list(existing):
        if tag.upper() in CATEGORY_NAMES:
            del existing[tag]
            cleaned += 1
            continue
        paths = existing.get(tag, [])
        if isinstance(paths, list):
            existing[tag] = sorted(set(str(path) for path in paths))
    if cleaned:
        LOGGER.info("Removed %d stale category tags from index", cleaned)

    ensure_dir(tag_index_path.parent)
    save_json(tag_index_path, existing)

    LOGGER.info("Tag index updated (tags=%s updates=%s)", len(existing), updates)
    print(f"Updated tag index with {updates} links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
