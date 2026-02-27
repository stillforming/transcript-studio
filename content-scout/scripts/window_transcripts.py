#!/usr/bin/env python3
"""Attach transcript windows to each extracted frame."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from _common import load_json, resolve_path, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.window_transcripts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", default="tmp/frames", help="Directory containing frame manifests")
    parser.add_argument(
        "--transcripts-dir",
        default="tmp/transcripts",
        help="Directory containing transcript JSON files",
    )
    parser.add_argument("--window", type=int, default=30, help="Window size in seconds around each frame")
    parser.add_argument("--output", default="tmp/windowed_frames.json", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def iter_manifests(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*/_manifest.json"))


def build_window_text(segments: list[dict[str, Any]], timestamp: float, window: int) -> str:
    lower = timestamp - window
    upper = timestamp + window
    snippets: list[str] = []

    for segment in segments:
        try:
            start = float(segment.get("start", 0.0))
        except (TypeError, ValueError):
            continue
        if lower <= start <= upper:
            text = str(segment.get("text") or "").strip()
            if text:
                snippets.append(text)
    return " ".join(snippets).strip()


def load_transcript(transcripts_dir: Path, video_id: str) -> dict[str, Any]:
    transcript_path = transcripts_dir / f"{video_id}.json"
    return load_json(transcript_path, default={})


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    frames_dir = resolve_path(args.frames_dir)
    transcripts_dir = resolve_path(args.transcripts_dir)
    output_path = resolve_path(args.output)

    output_rows: list[dict[str, Any]] = []

    manifests = iter_manifests(frames_dir)
    if not manifests:
        LOGGER.warning("No frame manifests found in %s", frames_dir)

    for manifest_path in manifests:
        video_id = manifest_path.parent.name
        manifest = load_json(manifest_path, default=[])
        transcript_payload = load_transcript(transcripts_dir, video_id)
        segments = transcript_payload.get("segments", [])
        if not isinstance(segments, list):
            segments = []

        video_title = str(transcript_payload.get("videoTitle") or "")
        channel_name = str(transcript_payload.get("channelName") or "")
        channel_slug = str(transcript_payload.get("channelSlug") or "")

        for item in manifest:
            timestamp = float(item.get("timestamp", 0.0))
            frame_path = str(item.get("path") or "")
            transcript_window = build_window_text(segments, timestamp, args.window)
            output_rows.append(
                {
                    "videoId": video_id,
                    "videoTitle": video_title,
                    "channelName": channel_name,
                    "channelSlug": channel_slug,
                    "framePath": frame_path,
                    "timestamp": int(timestamp),
                    "sourceUrl": f"https://youtube.com/watch?v={video_id}&t={int(timestamp)}",
                    "transcriptWindow": transcript_window,
                }
            )

    save_json(output_path, output_rows)
    LOGGER.info("Windowing complete (frames=%s output=%s)", len(output_rows), output_path)
    print(f"Windowed {len(output_rows)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
