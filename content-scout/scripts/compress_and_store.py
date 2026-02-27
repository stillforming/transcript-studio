#!/usr/bin/env python3
"""Compress selected frames and store them in the content vault."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from _common import ROOT_DIR, ensure_dir, load_json, normalize_slug, resolve_path, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.compress_and_store")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="tmp/annotations.json", help="Annotations JSON path")
    parser.add_argument("--frames-dir", default="tmp/frames", help="Base directory for frame paths")
    parser.add_argument("--output-dir", default="content-vault/daily", help="Output directory")
    parser.add_argument("--format", default="webp", help="Output image format")
    parser.add_argument("--quality", type=int, default=85, help="Output quality")
    parser.add_argument("--max-width", type=int, default=1920, help="Maximum output width")
    parser.add_argument("--max-per-video", type=int, default=25,
                        help="Maximum frames to store per video (top by hook_quality)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def resolve_frame(frame_path: str, frames_dir: Path) -> Path:
    candidate = Path(frame_path)
    if candidate.is_absolute():
        return candidate
    root_relative = ROOT_DIR / candidate
    if root_relative.exists():
        return root_relative
    return frames_dir / candidate


def compute_relative_to_vault(path: Path) -> str:
    vault_root = resolve_path("content-vault")
    try:
        return str(path.relative_to(vault_root))
    except ValueError:
        try:
            return str(path.relative_to(ROOT_DIR))
        except ValueError:
            return str(path)


def unique_output_path(output_dir: Path, filename: str) -> Path:
    candidate = output_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = output_dir / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def compress_image(source: Path, destination: Path, quality: int, max_width: int, fmt: str) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            scale = max_width / image.width
            new_size = (max_width, int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        image.save(destination, format=fmt.upper(), quality=quality, optimize=True)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    annotations_path = resolve_path(args.annotations)
    frames_dir = resolve_path(args.frames_dir)
    output_dir = ensure_dir(resolve_path(args.output_dir))

    annotations = load_json(annotations_path, default=[])
    if not isinstance(annotations, list):
        raise ValueError(f"Annotations must be a list: {annotations_path}")

    # Keep annotated frames, but skip storing TALKING_HEAD images (transcript is the value, not the image)
    kept_annotations = [
        item for item in annotations
        if isinstance(item, dict) and item.get("kept")
        and item.get("category") != "TALKING_HEAD"
    ]
    talking_heads = sum(
        1 for item in annotations
        if isinstance(item, dict) and item.get("kept") and item.get("category") == "TALKING_HEAD"
    )
    if talking_heads:
        LOGGER.info("Skipped %d TALKING_HEAD frames (transcript-only value)", talking_heads)

    # Cap frames per video to prevent long-video bias
    max_per = args.max_per_video
    if max_per and max_per > 0:
        from collections import defaultdict
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in kept_annotations:
            vid = str(item.get("videoId") or "unknown")
            by_video[vid].append(item)

        capped: list[dict[str, Any]] = []
        for vid, frames in by_video.items():
            if len(frames) <= max_per:
                capped.extend(frames)
            else:
                # Sort by hook_quality desc, then timestamp for tie-breaking
                sorted_frames = sorted(
                    frames,
                    key=lambda f: (-(f.get("hook_quality") or 0), f.get("timestamp") or 0),
                )
                capped.extend(sorted_frames[:max_per])
                LOGGER.info("Capped %s from %d to %d frames (by hook_quality)", vid, len(frames), max_per)

        # Re-sort by video + timestamp for chronological storage
        capped.sort(key=lambda f: (str(f.get("videoId") or ""), f.get("timestamp") or 0))
        trimmed = len(kept_annotations) - len(capped)
        if trimmed:
            LOGGER.info("Trimmed %d frames total across videos (max_per_video=%d)", trimmed, max_per)
        kept_annotations = capped
    stored_index: list[dict[str, Any]] = []
    stored = 0
    failed = 0

    for item in kept_annotations:
        try:
            frame_path = str(item.get("framePath") or "").strip()
            if not frame_path:
                raise ValueError("Missing framePath")

            source_path = resolve_frame(frame_path, frames_dir)
            if not source_path.exists():
                raise FileNotFoundError(source_path)

            channel_slug = normalize_slug(str(item.get("channelSlug") or item.get("channelName") or "channel"))
            video_id = str(item.get("videoId") or "video")
            timestamp = int(item.get("timestamp") or 0)
            extension = f".{args.format.lower()}"
            output_name = f"{channel_slug}_{video_id}_{timestamp}s{extension}"
            output_path = unique_output_path(output_dir, output_name)

            compress_image(source_path, output_path, args.quality, args.max_width, args.format)

            index_item = {
                "path": compute_relative_to_vault(output_path),
                "filename": output_path.name,
                "videoId": item.get("videoId"),
                "videoTitle": item.get("videoTitle"),
                "channelName": item.get("channelName"),
                "channelSlug": item.get("channelSlug"),
                "timestamp": item.get("timestamp"),
                "sourceUrl": item.get("sourceUrl"),
                "category": item.get("category"),
                "confidence": item.get("confidence"),
                "description": item.get("description"),
                "verbal_context": item.get("verbal_context"),
                "content_format": item.get("content_format"),
                "visual_technique": item.get("visual_technique"),
                "topic": item.get("topic"),
                "insight": item.get("insight"),
                "hook_quality": item.get("hook_quality"),
                "content_idea": item.get("content_idea"),
                "tags": item.get("tags", []),
                "ticker": item.get("ticker"),
                "transcriptWindow": item.get("transcriptWindow"),
            }

            stored_index.append(index_item)
            stored += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            LOGGER.exception("Failed to compress/store frame %s: %s", item.get("framePath"), exc)

    index_path = output_dir / "_index.json"
    save_json(index_path, stored_index)

    # Write per-video summary for audit trail (explains zero-yield videos)
    video_summary: dict[str, dict[str, Any]] = {}
    for item in annotations:
        if not isinstance(item, dict):
            continue
        vid = str(item.get("videoId") or "unknown")
        if vid not in video_summary:
            video_summary[vid] = {
                "videoId": vid,
                "videoTitle": item.get("videoTitle"),
                "channelName": item.get("channelName"),
                "channelSlug": item.get("channelSlug"),
                "totalFrames": 0,
                "keptFrames": 0,
                "storedFrames": 0,
                "categories": {},
                "avgConfidence": 0.0,
                "topHookQuality": 0,
            }
        summary = video_summary[vid]
        summary["totalFrames"] += 1
        cat = str(item.get("category") or "UNKNOWN")
        summary["categories"][cat] = summary["categories"].get(cat, 0) + 1
        if item.get("kept"):
            summary["keptFrames"] += 1
        hook = item.get("hook_quality")
        if isinstance(hook, (int, float)) and hook > summary["topHookQuality"]:
            summary["topHookQuality"] = int(hook)

    # Count stored frames per video from stored_index
    for entry in stored_index:
        vid = str(entry.get("videoId") or "unknown")
        if vid in video_summary:
            video_summary[vid]["storedFrames"] += 1

    # Compute average confidence per video
    for vid, summary in video_summary.items():
        vid_confs = [
            item.get("confidence", 0)
            for item in annotations
            if isinstance(item, dict) and str(item.get("videoId") or "") == vid
               and isinstance(item.get("confidence"), (int, float))
        ]
        if vid_confs:
            summary["avgConfidence"] = round(sum(vid_confs) / len(vid_confs), 3)

    summary_path = output_dir / "_summary.json"
    save_json(summary_path, list(video_summary.values()))

    LOGGER.info("Compression complete (stored=%s failed=%s index=%s)", stored, failed, index_path)
    print(f"Stored {stored} frames (failed {failed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
