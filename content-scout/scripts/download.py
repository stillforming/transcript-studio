#!/usr/bin/env python3
"""Download video and audio assets for selected videos."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from _common import ensure_dir, load_json, resolve_path, run_command, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tmp/video_list.json", help="Selected videos JSON path")
    parser.add_argument("--output-dir", default="tmp/downloads", help="Download directory")
    parser.add_argument("--max-resolution", type=int, default=720, help="Maximum video height")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds to sleep between videos")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def find_downloaded_file(output_dir: Path, prefix: str) -> str | None:
    for file_path in sorted(output_dir.glob(f"{prefix}.*")):
        if file_path.is_file():
            return file_path.name
    return None


def download_one(video: dict[str, Any], output_dir: Path, max_resolution: int, force: bool) -> dict[str, Any]:
    video_id = str(video.get("id") or "").strip()
    url = str(video.get("url") or f"https://youtube.com/watch?v={video_id}")
    if not video_id:
        raise ValueError("Video entry is missing 'id'")

    video_template = output_dir / f"{video_id}_video.%(ext)s"
    audio_template = output_dir / f"{video_id}_audio.%(ext)s"

    existing_video = find_downloaded_file(output_dir, f"{video_id}_video")
    existing_audio = output_dir / f"{video_id}_audio.mp3"

    if force or not existing_video:
        cmd_video = [
            "yt-dlp",
            "-f",
            f"bestvideo[height<={max_resolution}]",
            "-o",
            str(video_template),
            url,
        ]
        LOGGER.debug("Running: %s", " ".join(cmd_video))
        run_command(cmd_video)
        existing_video = find_downloaded_file(output_dir, f"{video_id}_video")

    if force or not existing_audio.exists():
        cmd_audio = [
            "yt-dlp",
            "-f",
            "bestaudio",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "-o",
            str(audio_template),
            url,
        ]
        LOGGER.debug("Running: %s", " ".join(cmd_audio))
        run_command(cmd_audio)

    if not existing_video:
        raise FileNotFoundError(f"Downloaded video file not found for {video_id}")
    if not existing_audio.exists():
        raise FileNotFoundError(f"Downloaded audio file not found for {video_id}")

    metadata = {
        "videoId": video_id,
        "url": url,
        "title": video.get("title", ""),
        "channelId": video.get("channelId", ""),
        "channelName": video.get("channelName", ""),
        "channelSlug": video.get("channelSlug", ""),
        "uploadDate": video.get("uploadDate"),
        "duration": video.get("duration"),
        "score": video.get("score"),
        "videoFile": existing_video,
        "audioFile": existing_audio.name,
    }

    save_json(output_dir / f"{video_id}_meta.json", metadata)
    return metadata


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    input_path = resolve_path(args.input)
    output_dir = ensure_dir(resolve_path(args.output_dir))

    videos = load_json(input_path, default=[])
    if not isinstance(videos, list):
        raise ValueError(f"Input file must be a list: {input_path}")

    downloaded = 0
    failed = 0

    for index, video in enumerate(videos):
        try:
            result = download_one(video, output_dir, args.max_resolution, args.force)
            downloaded += 1
            LOGGER.info(
                "Downloaded %s (%s)",
                result.get("videoId"),
                result.get("title", "").strip(),
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            LOGGER.exception("Failed to download entry %s: %s", index, exc)

        if index < len(videos) - 1 and args.delay > 0:
            time.sleep(args.delay)

    LOGGER.info("Download complete (downloaded=%s failed=%s)", downloaded, failed)
    print(f"Downloaded {downloaded} videos (failed {failed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
