#!/usr/bin/env python3
"""Merge transcript text segments with visual annotations."""

from __future__ import annotations

import argparse
import bisect
import logging
import os
from pathlib import Path
from typing import Any

from _common import ContentScoutError, ROOT_DIR, load_json, resolve_path, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.merge_visuals")

EXCLUDED_CATEGORIES = {"TALKING_HEAD", "FILLER"}
VISUAL_TYPE_MAP = {
    "CHART_VISUAL": "Chart",
    "SCREEN": "Screen",
    "SLIDE": "Slide",
    "GRAPHIC": "Graphic",
    "TABLE": "Table",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to {video_id}_transcript.json produced by transcribe_local.py",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations.json produced by classify_annotate.py",
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing frame image files",
    )
    parser.add_argument("--output", required=True, help="Path to write merged_transcript.json")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_timestamp(seconds: int) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def normalize_segments(payload: Any) -> list[dict[str, Any]]:
    raw_segments: Any
    if isinstance(payload, dict):
        raw_segments = payload.get("segments", [])
    elif isinstance(payload, list):
        raw_segments = payload
    else:
        raise ContentScoutError("Transcript file must contain an object or an array")

    if not isinstance(raw_segments, list):
        raise ContentScoutError("Transcript segments must be a list")

    normalized: list[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue

        start = max(0.0, coerce_float(raw.get("start"), 0.0))
        end = max(start, coerce_float(raw.get("end"), start))
        speaker = str(raw.get("speaker") or "Speaker 1")
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
                "minute_mark": raw.get("minute_mark"),
            }
        )

    normalized.sort(key=lambda segment: segment["start"])
    return normalized


def is_embed_candidate(item: dict[str, Any]) -> bool:
    if not item.get("kept"):
        return False
    category = str(item.get("category") or "").strip().upper()
    if not category:
        return False
    return category not in EXCLUDED_CATEGORIES


def normalize_visual_type(category: str) -> str:
    clean = str(category or "").strip().upper()
    if clean in VISUAL_TYPE_MAP:
        return VISUAL_TYPE_MAP[clean]
    if not clean:
        return "Visual"
    return clean.replace("_", " ").title()


def _as_absolute(candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def resolve_image_path(frame_ref: str, frames_dir: Path) -> str:
    frame_ref = str(frame_ref or "").strip()
    if not frame_ref:
        return ""

    candidate = Path(frame_ref)
    frames_dir_abs = _as_absolute(frames_dir).resolve()

    resolved: Path
    if candidate.is_absolute():
        resolved = candidate
    else:
        root_relative = ROOT_DIR / candidate
        frames_relative = frames_dir_abs / candidate
        if root_relative.exists():
            resolved = root_relative
        elif frames_relative.exists():
            resolved = frames_relative
        else:
            resolved = frames_relative

    resolved_abs = _as_absolute(resolved).resolve(strict=False)
    try:
        return resolved_abs.relative_to(frames_dir_abs).as_posix()
    except ValueError:
        pass

    try:
        return resolved_abs.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        pass

    return Path(os.path.relpath(resolved_abs, frames_dir_abs)).as_posix()


def normalize_visuals(annotations: list[dict[str, Any]], frames_dir: Path) -> list[dict[str, Any]]:
    visuals: list[dict[str, Any]] = []
    for source_index, item in enumerate(annotations):
        if not isinstance(item, dict) or not is_embed_candidate(item):
            continue

        raw_seconds = max(0.0, coerce_float(item.get("timestamp"), 0.0))
        timestamp_seconds = int(raw_seconds)
        category = str(item.get("category") or "").strip().upper()
        description = str(item.get("description") or "").strip()
        frame_ref = str(item.get("framePath") or item.get("frame") or "").strip()

        visuals.append(
            {
                "_timestamp_float": raw_seconds,
                "_source_index": source_index,
                "type": "visual",
                "timestamp_seconds": timestamp_seconds,
                "timestamp_label": format_timestamp(timestamp_seconds),
                "visual_type": normalize_visual_type(category),
                "description": description,
                "image_path": resolve_image_path(frame_ref, frames_dir),
            }
        )

    visuals.sort(key=lambda item: (item["_timestamp_float"], item["_source_index"]))
    return visuals


def filter_annotations_for_video(
    annotations: list[dict[str, Any]],
    video_id: str,
) -> list[dict[str, Any]]:
    if not video_id:
        return annotations

    with_video_id = [item for item in annotations if isinstance(item, dict) and item.get("videoId") is not None]
    if not with_video_id:
        return annotations

    filtered = [item for item in annotations if str(item.get("videoId") or "").strip() == video_id]
    return filtered


def assign_visuals_to_segments(
    segments: list[dict[str, Any]],
    visuals: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    mapping: dict[int, list[dict[str, Any]]] = {}
    starts = [segment["start"] for segment in segments]

    for visual in visuals:
        anchor = bisect.bisect_right(starts, visual["_timestamp_float"]) - 1
        clean_visual = {k: v for k, v in visual.items() if not k.startswith("_")}
        mapping.setdefault(anchor, []).append(clean_visual)

    return mapping


def build_event_sequence(
    segments: list[dict[str, Any]],
    visual_mapping: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for visual in visual_mapping.get(-1, []):
        events.append(visual)

    for index, segment in enumerate(segments):
        events.append(
            {
                "type": "text",
                "speaker": segment["speaker"],
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            }
        )
        for visual in visual_mapping.get(index, []):
            events.append(visual)

    return events


def event_time_seconds(event: dict[str, Any]) -> int:
    if event.get("type") == "text":
        return int(max(0.0, coerce_float(event.get("start"), 0.0)))
    return int(max(0.0, coerce_float(event.get("timestamp_seconds"), 0.0)))


def inject_timestamp_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []

    output: list[dict[str, Any]] = []
    next_boundary = 0

    for event in events:
        event_seconds = event_time_seconds(event)
        while next_boundary <= event_seconds:
            output.append({"type": "timestamp", "value": format_timestamp(next_boundary)})
            next_boundary += 60
        output.append(event)

    return output


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        transcript_path = resolve_path(args.transcript)
        annotations_path = resolve_path(args.annotations)
        frames_dir = resolve_path(args.frames_dir)
        output_path = resolve_path(args.output)

        transcript_payload = load_json(transcript_path, default={})
        annotations = load_json(annotations_path, default=[])

        if not isinstance(annotations, list):
            raise ContentScoutError(f"Annotations must be a list: {annotations_path}")

        transcript_video_id = str(
            transcript_payload.get("videoId") or transcript_path.stem.replace("_transcript", "")
        ).strip()
        scoped_annotations = filter_annotations_for_video(annotations, transcript_video_id)

        segments = normalize_segments(transcript_payload)
        visuals = normalize_visuals(scoped_annotations, frames_dir)
        visual_mapping = assign_visuals_to_segments(segments, visuals)
        events = build_event_sequence(segments, visual_mapping)
        merged_blocks = inject_timestamp_blocks(events)

        save_json(output_path, merged_blocks)
        LOGGER.info(
            "Merged transcript written to %s (video=%s segments=%s embed_visuals=%s blocks=%s)",
            output_path,
            transcript_video_id or "unknown",
            len(segments),
            len(visuals),
            len(merged_blocks),
        )
        print(f"Merged {len(segments)} segments with {len(visuals)} visuals into {output_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("merge_visuals failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
