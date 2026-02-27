#!/usr/bin/env python3
"""Update processing log with processed video IDs and daily metrics."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Any

from _common import load_json, resolve_path, save_json, setup_logging, utc_today_str, utcnow

LOGGER = logging.getLogger("content_scout.update_log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-list", default="tmp/video_list.json", help="Selected videos JSON path")
    parser.add_argument("--annotations", default="tmp/annotations.json", help="Annotations JSON path")
    parser.add_argument("--log", default="content-vault/processing-log.json", help="Processing log path")
    parser.add_argument("--date", help="Override stats date (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def ensure_log_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("processedVideoIds", [])
    payload.setdefault("dailyStats", {})
    return payload


def normalize_date(date_str: str | None) -> str:
    if not date_str:
        return utc_today_str()
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    video_list_path = resolve_path(args.video_list)
    annotations_path = resolve_path(args.annotations)
    log_path = resolve_path(args.log)

    date_key = normalize_date(args.date)

    videos = load_json(video_list_path, default=[])
    annotations = load_json(annotations_path, default=[])
    log_payload = ensure_log_shape(load_json(log_path, default={}))

    if not isinstance(videos, list):
        raise ValueError(f"Video list must be a list: {video_list_path}")
    if not isinstance(annotations, list):
        raise ValueError(f"Annotations must be a list: {annotations_path}")

    existing_ids = set(str(item) for item in log_payload.get("processedVideoIds", []))
    selected_ids = {str(item.get("id") or item.get("videoId") or "") for item in videos}
    selected_ids = {video_id for video_id in selected_ids if video_id}

    existing_ids.update(selected_ids)
    log_payload["processedVideoIds"] = sorted(existing_ids)

    kept_annotations = [item for item in annotations if isinstance(item, dict) and item.get("kept")]
    channels = {
        str(item.get("channelSlug") or item.get("channelName") or "").strip()
        for item in videos
        if isinstance(item, dict)
    }
    channels.discard("")

    tickers = {
        str(item.get("ticker") or "").strip()
        for item in kept_annotations
        if isinstance(item, dict)
    }
    tickers.discard("")

    daily_stats = log_payload.setdefault("dailyStats", {})
    stats_for_day = daily_stats.get(date_key, {}) if isinstance(daily_stats, dict) else {}
    if not isinstance(stats_for_day, dict):
        stats_for_day = {}

    stats_for_day.update(
        {
            "videosSelected": len(videos),
            "videosProcessed": len(selected_ids),
            "framesAnnotated": len(annotations),
            "framesKept": len(kept_annotations),
            "channels": sorted(channels),
            "tickers": sorted(tickers),
            "updatedAt": utcnow().isoformat(),
        }
    )

    daily_stats[date_key] = stats_for_day
    log_payload["dailyStats"] = daily_stats

    save_json(log_path, log_payload)
    LOGGER.info(
        "Processing log updated (date=%s selected=%s kept=%s)",
        date_key,
        len(videos),
        len(kept_annotations),
    )
    print(f"Updated processing log for {date_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
