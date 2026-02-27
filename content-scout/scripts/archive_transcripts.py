#!/usr/bin/env python3
"""Archive transcripts and collect pipeline errors before cleanup.

Copies transcript JSON files from tmp/ to content-vault/transcripts/{date}/
and aggregates any errors from the pipeline run into content-vault/errors/{date}.json.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from _common import ensure_dir, load_json, resolve_path, save_json, setup_logging, utcnow

LOGGER = logging.getLogger("content_scout.archive_transcripts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-dir", default="tmp/transcripts",
                        help="Source directory with transcript JSON files")
    parser.add_argument("--output-dir", default="content-vault/transcripts",
                        help="Destination directory for archived transcripts")
    parser.add_argument("--annotations", default="tmp/annotations.json",
                        help="Annotations file (to extract classification errors)")
    parser.add_argument("--errors-dir", default="content-vault/errors",
                        help="Directory for error log files")
    parser.add_argument("--date", default=None,
                        help="Pipeline date (YYYY-MM-DD) for error log filename")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def archive_transcripts(src_dir: Path, dst_dir: Path) -> int:
    """Copy transcript JSON files to the archive directory. Returns count."""
    if not src_dir.exists():
        LOGGER.info("No transcripts directory found at %s — skipping", src_dir)
        return 0

    ensure_dir(dst_dir)
    copied = 0

    for json_file in sorted(src_dir.glob("*.json")):
        dest = dst_dir / json_file.name
        shutil.copy2(json_file, dest)
        copied += 1
        LOGGER.debug("Archived %s → %s", json_file.name, dest)

    return copied


def archive_annotations(annotations_path: Path, dst_dir: Path) -> bool:
    """Copy annotations.json to the archive directory. Returns True if copied."""
    if not annotations_path.exists():
        LOGGER.info("No annotations file at %s — skipping", annotations_path)
        return False

    ensure_dir(dst_dir)
    dest = dst_dir / "annotations.json"
    shutil.copy2(annotations_path, dest)
    LOGGER.debug("Archived annotations → %s", dest)
    return True


def collect_errors(annotations_path: Path, pipeline_state_path: Path | None) -> list[dict[str, Any]]:
    """Collect errors from annotations (failed classifications) and pipeline state."""
    errors: list[dict[str, Any]] = []

    # Classification errors: frames that weren't kept due to low confidence or FILLER
    annotations = load_json(annotations_path, default=[])
    if isinstance(annotations, list):
        for item in annotations:
            if not isinstance(item, dict):
                continue
            # Frames with explicit errors or that failed classification
            if item.get("category") == "FILLER" and item.get("confidence", 0) == 0:
                errors.append({
                    "type": "classification_failure",
                    "videoId": item.get("videoId"),
                    "framePath": item.get("framePath"),
                    "timestamp": item.get("timestamp"),
                    "reason": "Zero confidence — model may not have returned annotation for this frame",
                    "collectedAt": utcnow().isoformat(),
                })

    # Transcript errors: files with error field
    # (these are already archived by this point)
    # We don't re-read them here; the transcribe step logs its own failures.

    # Pipeline state errors
    if pipeline_state_path and pipeline_state_path.exists():
        state = load_json(pipeline_state_path, default={})
        steps = state.get("steps", {})
        for step_name, step_data in steps.items():
            if isinstance(step_data, dict) and step_data.get("status") == "failed":
                errors.append({
                    "type": "step_failure",
                    "step": step_name,
                    "message": step_data.get("message", ""),
                    "updatedAt": step_data.get("updatedAt"),
                    "collectedAt": utcnow().isoformat(),
                })

    return errors


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    src_dir = resolve_path(args.transcripts_dir)
    dst_dir = resolve_path(args.output_dir)
    annotations_path = resolve_path(args.annotations)
    errors_dir = resolve_path(args.errors_dir)
    pipeline_state = resolve_path("tmp/_pipeline_state.json")

    # 1. Archive transcripts
    copied = archive_transcripts(src_dir, dst_dir)
    LOGGER.info("Archived %d transcript files to %s", copied, dst_dir)

    # 1b. Archive annotations (expensive to regenerate — ~21 min classify step)
    archived_ann = archive_annotations(annotations_path, dst_dir)
    if archived_ann:
        LOGGER.info("Archived annotations.json to %s", dst_dir)

    # 2. Collect and write errors
    errors = collect_errors(annotations_path, pipeline_state)

    if errors:
        ensure_dir(errors_dir)
        date_str = args.date or utcnow().date().isoformat()
        error_path = errors_dir / f"{date_str}.json"

        # Merge with existing errors for the same date
        existing = load_json(error_path, default=[])
        if not isinstance(existing, list):
            existing = []
        existing.extend(errors)

        save_json(error_path, existing)
        LOGGER.info("Wrote %d errors to %s", len(errors), error_path)
    else:
        LOGGER.info("No errors to record")

    print(f"Archived {copied} transcripts, recorded {len(errors)} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
