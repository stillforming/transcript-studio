#!/usr/bin/env python3
"""Extract and de-duplicate video frames using perceptual hash distance."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from _common import ROOT_DIR, ensure_dir, load_json, resolve_path, run_command, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.extract_frames")
FRAME_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="tmp/downloads", help="Directory containing downloaded assets")
    parser.add_argument("--output-dir", default="tmp/frames", help="Directory to write extracted frames")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between extracted frames")
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=5,
        help="Minimum Hamming distance between kept frames",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def parse_frame_number(frame_name: str) -> int:
    match = FRAME_RE.search(frame_name)
    if not match:
        return 0
    return int(match.group(1))


def rel_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def extract_video_frames(video_file: Path, raw_dir: Path, interval: int) -> list[Path]:
    ensure_dir(raw_dir)
    frame_pattern = raw_dir / "frame_%04d.png"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_file),
        "-vf",
        f"fps=1/{interval}",
        "-q:v",
        "2",
        str(frame_pattern),
    ]
    LOGGER.debug("Running: %s", " ".join(cmd))
    run_command(cmd)
    return sorted(raw_dir.glob("frame_*.png"))


def deduplicate_frames(
    raw_frames: list[Path],
    output_video_dir: Path,
    interval: int,
    threshold: int,
) -> list[dict[str, Any]]:
    kept_hashes: list[imagehash.ImageHash] = []
    manifest: list[dict[str, Any]] = []

    for frame_path in raw_frames:
        with Image.open(frame_path) as image:
            frame_hash = imagehash.phash(image)

        is_duplicate = any((frame_hash - existing) < threshold for existing in kept_hashes)
        if is_duplicate:
            continue

        frame_number = parse_frame_number(frame_path.stem)
        timestamp = frame_number * interval
        final_frame = output_video_dir / f"frame_{timestamp}s.png"
        shutil.move(str(frame_path), final_frame)

        kept_hashes.append(frame_hash)
        manifest.append(
            {
                "path": rel_str(final_frame),
                "timestamp": timestamp,
                "hash": str(frame_hash),
            }
        )

    return manifest


def iter_meta_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_meta.json"))


def resolve_video_file(input_dir: Path, video_id: str, metadata: dict[str, Any]) -> Path | None:
    meta_video = metadata.get("videoFile")
    if isinstance(meta_video, str) and meta_video:
        candidate = input_dir / meta_video
        if candidate.exists():
            return candidate

    matches = sorted(input_dir.glob(f"{video_id}_video.*"))
    for match in matches:
        if match.is_file():
            return match
    return None


def process_video(meta_path: Path, input_dir: Path, output_dir: Path, interval: int, threshold: int) -> int:
    metadata = load_json(meta_path, default={})
    video_id = str(metadata.get("videoId") or meta_path.stem.replace("_meta", ""))
    video_file = resolve_video_file(input_dir, video_id, metadata)
    if video_file is None:
        LOGGER.warning("Skipping %s because video file is missing", video_id)
        return 0

    output_video_dir = ensure_dir(output_dir / video_id)
    raw_dir = ensure_dir(output_video_dir / "_raw")

    # Clean previous artifacts for deterministic reruns.
    for old_frame in output_video_dir.glob("frame_*s.png"):
        old_frame.unlink(missing_ok=True)

    raw_frames = extract_video_frames(video_file, raw_dir, interval)
    manifest = deduplicate_frames(raw_frames, output_video_dir, interval, threshold)

    save_json(output_video_dir / "_manifest.json", manifest)

    shutil.rmtree(raw_dir, ignore_errors=True)
    return len(manifest)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    input_dir = resolve_path(args.input_dir)
    output_dir = ensure_dir(resolve_path(args.output_dir))

    total_kept = 0
    failures = 0
    metas = iter_meta_files(input_dir)

    if not metas:
        LOGGER.warning("No metadata files found in %s", input_dir)

    for meta_path in metas:
        try:
            kept = process_video(meta_path, input_dir, output_dir, args.interval, args.hash_threshold)
            total_kept += kept
            LOGGER.info("Processed %s -> kept %s frames", meta_path.name, kept)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            LOGGER.exception("Frame extraction failed for %s: %s", meta_path.name, exc)

    LOGGER.info("Frame extraction complete (kept=%s failures=%s)", total_kept, failures)
    print(f"Extracted {total_kept} unique frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
