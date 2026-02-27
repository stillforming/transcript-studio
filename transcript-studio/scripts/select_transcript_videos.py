#!/usr/bin/env python3
"""Select Transcript Studio videos from configured channels and playlists."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

from _common import (
    load_json,
    normalize_slug,
    parse_upload_date,
    resolve_path,
    run_command,
    save_json,
    setup_logging,
)

LOGGER = logging.getLogger("content_scout.select_transcript_videos")
DEFAULT_PRESET = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels",
        default="config/channels.json",
        help="Path to channels JSON config",
    )
    parser.add_argument(
        "--playlists",
        default="config/playlists.json",
        help="Path to playlists JSON config",
    )
    parser.add_argument(
        "--log",
        default="content-vault/transcript-studio-log.json",
        help="Path to processing log JSON",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum videos to output")
    parser.add_argument(
        "--playlist-end",
        type=int,
        default=10,
        help="How many recent uploads to inspect per channel",
    )
    parser.add_argument("--output", default="tmp/ts_video_list.json", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def load_processed_ids(log_path_str: str) -> set[str]:
    payload = load_json(resolve_path(log_path_str), default={})
    processed = payload.get("processedVideoIds", []) if isinstance(payload, dict) else []
    if not isinstance(processed, list):
        return set()
    return {str(video_id).strip() for video_id in processed if str(video_id).strip()}


def fetch_channel_entries(channel_url: str, playlist_end: int) -> list[dict[str, Any]]:
    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        url += "/videos"

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-end",
        str(max(1, playlist_end)),
        "--dump-single-json",
        url,
    ]
    LOGGER.debug("Running: %s", " ".join(cmd))
    completed = run_command(cmd, check=True)
    payload = json.loads(completed.stdout or "{}")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def fetch_playlist_entries(playlist_url: str) -> list[dict[str, Any]]:
    cmd = ["yt-dlp", "--flat-playlist", "--dump-single-json", playlist_url]
    LOGGER.debug("Running: %s", " ".join(cmd))
    completed = run_command(cmd, check=True)
    payload = json.loads(completed.stdout or "{}")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_upload_datetime(entry: dict[str, Any]) -> datetime | None:
    epoch = entry.get("timestamp") or entry.get("release_timestamp")
    if isinstance(epoch, (int, float)):
        return datetime.fromtimestamp(float(epoch), tz=UTC)
    raw_date = entry.get("upload_date") or entry.get("release_date")
    if isinstance(raw_date, str):
        return parse_upload_date(raw_date)
    return None


def normalize_channel_name(entry: dict[str, Any], fallback: str = "Unknown Channel") -> str:
    for key in ("channel", "uploader", "channel_name"):
        value = str(entry.get(key) or "").strip()
        if value:
            return value
    return fallback


def normalize_preset(config_item: dict[str, Any]) -> str:
    preset = str(config_item.get("preset") or DEFAULT_PRESET).strip()
    return preset or DEFAULT_PRESET


def build_channel_candidate(channel: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    video_id = str(entry.get("id") or "").strip()
    channel_name = str(channel.get("name") or "").strip() or normalize_channel_name(entry)
    channel_slug = str(channel.get("channelSlug") or channel.get("slug") or "").strip()
    if not channel_slug:
        channel_slug = normalize_slug(channel_name)

    upload_dt = extract_upload_datetime(entry)
    title = str(entry.get("title") or "").strip()
    duration = max(0, as_int(entry.get("duration"), 0))
    channel_id = str(channel.get("channelId") or channel.get("id") or entry.get("channel_id") or "").strip()

    return {
        "id": video_id,
        "url": f"https://youtube.com/watch?v={video_id}",
        "title": title,
        "channelId": channel_id,
        "channelName": channel_name,
        "channelSlug": channel_slug,
        "uploadDate": upload_dt.isoformat() if upload_dt else None,
        "duration": duration,
        "source": "channel",
        "playlistName": None,
        "preset": normalize_preset(channel),
    }


def build_playlist_candidate(playlist: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    video_id = str(entry.get("id") or "").strip()
    title = str(entry.get("title") or "").strip()
    channel_name = normalize_channel_name(entry)
    channel_slug = normalize_slug(channel_name)
    upload_dt = extract_upload_datetime(entry)
    duration = max(0, as_int(entry.get("duration"), 0))
    playlist_name = str(playlist.get("name") or "Unknown Playlist").strip()

    return {
        "id": video_id,
        "url": f"https://youtube.com/watch?v={video_id}",
        "title": title,
        "channelId": str(entry.get("channel_id") or "").strip(),
        "channelName": channel_name,
        "channelSlug": channel_slug,
        "uploadDate": upload_dt.isoformat() if upload_dt else None,
        "duration": duration,
        "source": "playlist",
        "playlistName": playlist_name,
        "preset": normalize_preset(playlist),
    }


def merge_candidate(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)

    for key, value in incoming.items():
        if key == "duration":
            incoming_duration = max(0, as_int(value, 0))
            existing_duration = max(0, as_int(existing.get("duration"), 0))
            if incoming_duration > 0 or existing_duration <= 0:
                merged[key] = incoming_duration
            continue

        if value is None:
            if key not in merged:
                merged[key] = None
            continue

        text_value = str(value).strip() if isinstance(value, str) else value
        if isinstance(text_value, str) and not text_value:
            if key not in merged:
                merged[key] = text_value
            continue
        merged[key] = text_value

    return merged


def load_items(payload: Any, root_key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get(root_key, [])
    else:
        raw_items = []
    if not isinstance(raw_items, list):
        return []
    return [
        item
        for item in raw_items
        if isinstance(item, dict) and item.get("status", "active") == "active"
    ]


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    channels_payload = load_json(resolve_path(args.channels), default={})
    playlists_payload = load_json(resolve_path(args.playlists), default={})
    processed_ids = load_processed_ids(args.log)

    channels = load_items(channels_payload, "channels")
    playlists = load_items(playlists_payload, "playlists")

    selected_by_id: dict[str, dict[str, Any]] = {}
    selected_order: list[str] = []

    for channel in channels:
        channel_name = str(channel.get("name") or "Unknown Channel")
        channel_url = str(channel.get("url") or "").strip()
        if not channel_url:
            LOGGER.warning("Skipping channel %s without URL", channel_name)
            continue

        try:
            entries = fetch_channel_entries(channel_url, args.playlist_end)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to fetch channel %s: %s", channel_name, exc)
            continue

        for entry in entries:
            video_id = str(entry.get("id") or "").strip()
            if not video_id or video_id in processed_ids:
                continue

            candidate = build_channel_candidate(channel, entry)
            if not candidate["title"]:
                continue

            if video_id not in selected_by_id:
                selected_by_id[video_id] = candidate
                selected_order.append(video_id)
            else:
                selected_by_id[video_id] = merge_candidate(selected_by_id[video_id], candidate)

    for playlist in playlists:
        playlist_name = str(playlist.get("name") or "Unknown Playlist")
        playlist_url = str(playlist.get("url") or "").strip()
        if not playlist_url:
            LOGGER.warning("Skipping playlist %s without URL", playlist_name)
            continue

        try:
            entries = fetch_playlist_entries(playlist_url)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to fetch playlist %s: %s", playlist_name, exc)
            continue

        for entry in entries:
            video_id = str(entry.get("id") or "").strip()
            if not video_id or video_id in processed_ids:
                continue

            candidate = build_playlist_candidate(playlist, entry)
            if not candidate["title"]:
                continue

            if video_id not in selected_by_id:
                selected_by_id[video_id] = candidate
                selected_order.append(video_id)
            else:
                # Playlist membership is an explicit signal; prefer playlist metadata.
                merged = merge_candidate(selected_by_id[video_id], candidate)
                merged["source"] = "playlist"
                merged["playlistName"] = candidate.get("playlistName")
                merged["preset"] = candidate.get("preset") or merged.get("preset") or DEFAULT_PRESET
                selected_by_id[video_id] = merged

    output: list[dict[str, Any]] = []
    for video_id in selected_order:
        candidate = selected_by_id.get(video_id)
        if not candidate:
            continue
        output.append(candidate)

    limited = output[: max(0, args.limit)]
    save_json(resolve_path(args.output), limited)

    LOGGER.info(
        "Selected %s videos (channels=%s playlists=%s output=%s)",
        len(limited),
        len(channels),
        len(playlists),
        args.output,
    )
    print(f"Selected {len(limited)} transcript videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
