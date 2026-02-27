#!/usr/bin/env python3
"""Select recently uploaded videos from configured channels."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common import (
    load_json,
    normalize_slug,
    parse_upload_date,
    resolve_path,
    run_command,
    save_json,
    setup_logging,
    utcnow,
)

LOGGER = logging.getLogger("content_scout.select_videos")
PRIORITY_SCORES = {"high": 1.0, "medium": 0.6, "low": 0.3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels",
        default="config/channels.json",
        help="Path to channels JSON config",
    )
    parser.add_argument(
        "--keywords",
        default="config/keywords.json",
        help="Path to keywords JSON config",
    )
    parser.add_argument(
        "--log",
        default="content-vault/processing-log.json",
        help="Path to processing log JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum videos to output",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=180,
        help="Minimum duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=3600,
        help="Maximum duration in seconds",
    )
    parser.add_argument(
        "--playlist-end",
        type=int,
        default=10,
        help="How many recent uploads to inspect per channel",
    )
    parser.add_argument("--output", default="tmp/video_list.json", help="Output JSON path")
    parser.add_argument("--channels-dir", default="content-vault/channels",
                        help="Directory to write per-channel scan metadata")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def load_processed_ids(log_path: Path) -> set[str]:
    payload = load_json(log_path, default={})
    processed = payload.get("processedVideoIds", [])
    return {str(video_id) for video_id in processed}


def fetch_channel_entries(channel_url: str, playlist_end: int) -> list[dict[str, Any]]:
    # Ensure we hit the /videos tab, not the channel root (which returns tab listings)
    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        url += "/videos"

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-end",
        str(playlist_end),
        "--dump-single-json",
        url,
    ]
    LOGGER.debug("Running: %s", " ".join(cmd))
    completed = run_command(cmd, check=True)
    payload = json.loads(completed.stdout or "{}")
    entries = payload.get("entries", [])
    return [entry for entry in entries if isinstance(entry, dict)]


def extract_upload_datetime(entry: dict[str, Any]) -> datetime | None:
    epoch = entry.get("timestamp") or entry.get("release_timestamp")
    if isinstance(epoch, (int, float)):
        return datetime.fromtimestamp(float(epoch), tz=UTC)
    raw_date = entry.get("upload_date") or entry.get("release_date")
    if isinstance(raw_date, str):
        return parse_upload_date(raw_date)
    return None


def title_contains_any(title: str, phrases: list[str]) -> bool:
    title_lc = title.lower()
    return any(phrase.lower() in title_lc for phrase in phrases)


def collect_keyword_score(title: str, high_keywords: list[str], medium_keywords: list[str]) -> tuple[float, list[str]]:
    title_lc = title.lower()
    score_points = 0
    matched: list[str] = []
    for keyword in high_keywords:
        if keyword.lower() in title_lc:
            score_points += 3
            matched.append(keyword)
    for keyword in medium_keywords:
        if keyword.lower() in title_lc:
            score_points += 1
            matched.append(keyword)
    keyword_score = min(score_points / 10.0, 1.0)
    return keyword_score, matched


def score_video(
    hours_since_upload: float,
    priority: str,
    keyword_score: float,
    recency_weight: float,
    priority_weight: float,
    keyword_weight: float,
) -> float:
    recency = max(0.0, 24.0 - hours_since_upload) / 24.0
    priority_component = PRIORITY_SCORES.get(priority.lower(), PRIORITY_SCORES["low"])
    return (
        recency * recency_weight
        + priority_component * priority_weight
        + keyword_score * keyword_weight
    )


def update_channel_metadata(channels: list[dict[str, Any]], scan_results: dict[str, dict[str, Any]],
                            channels_dir_path: str) -> None:
    """Write per-channel scan metadata to content-vault/channels/."""
    channels_dir = resolve_path(channels_dir_path)
    from _common import ensure_dir, save_json
    ensure_dir(channels_dir)

    for channel in channels:
        channel_name = str(channel.get("name") or "Unknown")
        channel_slug = str(channel.get("slug") or normalize_slug(channel_name))
        scan_data = scan_results.get(channel_slug, {})

        meta_path = channels_dir / f"{channel_slug}.json"

        # Merge with existing metadata
        existing = load_json(meta_path, default={})
        if not isinstance(existing, dict):
            existing = {}

        existing.update({
            "name": channel_name,
            "slug": channel_slug,
            "url": channel.get("url"),
            "handle": channel.get("handle"),
            "category": channel.get("category"),
            "priority": channel.get("priority"),
            "source": channel.get("source"),
            "lastScanned": utcnow().isoformat(),
            "lastScanVideosFound": scan_data.get("total_found", 0),
            "lastScanVideosSelected": scan_data.get("selected", 0),
            "lastScanError": scan_data.get("error"),
        })

        # Track scan history (keep last 30 entries)
        history = existing.get("scanHistory", [])
        if not isinstance(history, list):
            history = []
        history.append({
            "date": utcnow().isoformat(),
            "found": scan_data.get("total_found", 0),
            "selected": scan_data.get("selected", 0),
            "error": scan_data.get("error"),
        })
        existing["scanHistory"] = history[-30:]

        save_json(meta_path, existing)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    channels_path = resolve_path(args.channels)
    keywords_path = resolve_path(args.keywords)
    log_path = resolve_path(args.log)
    output_path = resolve_path(args.output)

    channels_payload = load_json(channels_path, default={})
    keywords_payload = load_json(keywords_path, default={})

    # Support both {"channels": [...]} and plain [...] formats
    if isinstance(channels_payload, list):
        channels_list = channels_payload
    else:
        channels_list = channels_payload.get("channels", [])

    channels = [
        channel
        for channel in channels_list
        if isinstance(channel, dict) and channel.get("status", "active") == "active"
    ]

    high_keywords = keywords_payload.get("highRelevance", [])
    medium_keywords = keywords_payload.get("mediumRelevance", [])
    exclude_keywords = keywords_payload.get("exclude", [])

    scoring_payload = load_json(resolve_path("config/settings.json"), default={})
    scoring = scoring_payload.get("scoring", {})
    recency_weight = float(scoring.get("recencyWeight", 0.3))
    priority_weight = float(scoring.get("priorityWeight", 0.3))
    keyword_weight = float(scoring.get("keywordWeight", 0.4))

    processed_ids = load_processed_ids(log_path)
    now = utcnow()

    selected: list[dict[str, Any]] = []
    scanned = 0
    channel_scan_results: dict[str, dict[str, Any]] = {}

    for channel in channels:
        channel_id = str(channel.get("id") or "")
        channel_name = str(channel.get("name") or channel_id or "Unknown Channel")
        channel_slug = str(channel.get("slug") or normalize_slug(channel_name))
        channel_priority = str(channel.get("priority", "low")).lower()
        channel_url = str(channel.get("url") or "").strip()

        if not channel_url:
            LOGGER.warning("Skipping channel %s without URL", channel_name)
            continue

        try:
            entries = fetch_channel_entries(channel_url, args.playlist_end)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to fetch channel %s: %s", channel_name, exc)
            channel_scan_results[channel_slug] = {
                "total_found": 0, "selected": 0, "error": str(exc),
            }
            continue

        channel_selected_count = 0

        for position, entry in enumerate(entries):
            scanned += 1
            video_id = str(entry.get("id") or "")
            title = str(entry.get("title") or "").strip()
            duration = int(entry.get("duration") or 0)

            if not video_id or not title:
                continue
            if video_id in processed_ids:
                continue
            # Duration 0 means flat-playlist didn't return it; allow through
            if duration > 0 and (duration < args.min_duration or duration > args.max_duration):
                continue
            if title_contains_any(title, exclude_keywords):
                continue

            upload_dt = extract_upload_datetime(entry)
            if upload_dt is not None:
                hours_since_upload = (now - upload_dt).total_seconds() / 3600
                if hours_since_upload < 0:
                    continue
                upload_iso = upload_dt.isoformat()
            else:
                # No timestamp available — use playlist position as recency proxy
                # Position 0 = most recent, assume ~12h per position as rough estimate
                hours_since_upload = float(position) * 12.0
                upload_iso = None

            keyword_score, _matched_keywords = collect_keyword_score(
                title,
                list(high_keywords),
                list(medium_keywords),
            )
            score = score_video(
                hours_since_upload=hours_since_upload,
                priority=channel_priority,
                keyword_score=keyword_score,
                recency_weight=recency_weight,
                priority_weight=priority_weight,
                keyword_weight=keyword_weight,
            )

            selected.append(
                {
                    "id": video_id,
                    "url": f"https://youtube.com/watch?v={video_id}",
                    "title": title,
                    "channelId": channel_id,
                    "channelName": channel_name,
                    "channelSlug": channel_slug,
                    "uploadDate": upload_iso,
                    "duration": duration,
                    "score": round(score, 6),
                }
            )
            channel_selected_count += 1

        channel_scan_results[channel_slug] = {
            "total_found": len(entries),
            "selected": channel_selected_count,
            "error": None,
        }

    selected.sort(key=lambda item: item["score"], reverse=True)
    output = selected[: max(args.limit, 0)]

    save_json(output_path, output)

    # Update per-channel metadata in content-vault/channels/
    try:
        update_channel_metadata(channels, channel_scan_results, args.channels_dir)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to update channel metadata: %s", exc)

    LOGGER.info(
        "Selected %s videos (scanned=%s channels=%s output=%s)",
        len(output),
        scanned,
        len(channels),
        output_path,
    )
    print(f"Selected {len(output)} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
