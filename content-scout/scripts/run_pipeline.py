#!/usr/bin/env python3
"""Run the full Content Scout pipeline with resume support."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from _common import (
    ensure_dir,
    load_json,
    normalize_slug,
    parse_upload_date,
    resolve_path,
    run_command,
    save_json,
    setup_logging,
    utc_today_str,
    utcnow,
)

LOGGER = logging.getLogger("content_scout.run_pipeline")


@dataclass(frozen=True)
class StepSpec:
    name: str
    script: str
    args: list[str]
    optional: bool = False


STEP_ORDER = [
    "select",
    "download",
    "extract",
    "transcribe",
    "window",
    "classify",
    "compress",
    "tags",
    "log",
    "notion",
    "brief",
    "archive",
    "cleanup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Pipeline date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--video-url", help="Single YouTube URL to process")
    parser.add_argument("--dry-run", action="store_true", help="Print steps without executing")
    parser.add_argument("--force", action="store_true", help="Ignore previous state and rerun all steps")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def normalize_date(date_str: str | None) -> str:
    if not date_str:
        return utc_today_str()
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def default_state() -> dict[str, Any]:
    return {
        "createdAt": utcnow().isoformat(),
        "updatedAt": utcnow().isoformat(),
        "steps": {step: {"status": "pending"} for step in STEP_ORDER},
    }


def load_state(state_path: Path, pipeline_date: str, video_url: str | None, force: bool) -> dict[str, Any]:
    if force or not state_path.exists():
        state = default_state()
        state["date"] = pipeline_date
        state["videoUrl"] = video_url
        return state

    state = load_json(state_path, default={})
    if not isinstance(state, dict):
        state = default_state()

    if state.get("date") != pipeline_date or state.get("videoUrl") != video_url:
        LOGGER.info("Existing state does not match request, creating a new state")
        state = default_state()

    state.setdefault("date", pipeline_date)
    state.setdefault("videoUrl", video_url)
    state.setdefault("steps", {step: {"status": "pending"} for step in STEP_ORDER})

    for step in STEP_ORDER:
        state["steps"].setdefault(step, {"status": "pending"})

    return state


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updatedAt"] = utcnow().isoformat()
    save_json(state_path, state)


def script_path(script_name: str) -> str:
    return str((Path(__file__).resolve().parent / script_name).resolve())


def build_steps(pipeline_date: str, run_id: str | None = None) -> list[StepSpec]:
    daily_dir = f"content-vault/daily/{pipeline_date}"
    # Human-readable brief name: _brief-2026-02-24-14h30.md
    brief_name = f"_brief-{pipeline_date}-{run_id}.md" if run_id else f"_brief-{pipeline_date}.md"

    return [
        StepSpec(
            name="select",
            script="select_videos.py",
            args=[
                "--channels",
                "config/channels.json",
                "--keywords",
                "config/keywords.json",
                "--log",
                "content-vault/processing-log.json",
                "--limit",
                "5",
                "--output",
                "tmp/video_list.json",
            ],
        ),
        StepSpec(
            name="download",
            script="download.py",
            args=[
                "--input",
                "tmp/video_list.json",
                "--output-dir",
                "tmp/downloads",
                "--max-resolution",
                "720",
                "--delay",
                "5",
            ],
        ),
        StepSpec(
            name="extract",
            script="extract_frames.py",
            args=[
                "--input-dir",
                "tmp/downloads",
                "--output-dir",
                "tmp/frames",
                "--interval",
                "15",
                "--hash-threshold",
                "5",
            ],
        ),
        StepSpec(
            name="transcribe",
            script="transcribe.py",
            args=[
                "--input-dir",
                "tmp/downloads",
                "--output-dir",
                "tmp/transcripts",
                "--model",
                "whisper-1",
            ],
            optional=True,
        ),
        StepSpec(
            name="window",
            script="window_transcripts.py",
            args=[
                "--frames-dir",
                "tmp/frames",
                "--transcripts-dir",
                "tmp/transcripts",
                "--window",
                "30",
                "--output",
                "tmp/windowed_frames.json",
            ],
        ),
        StepSpec(
            name="classify",
            script="classify_annotate.py",
            args=[
                "--input",
                "tmp/windowed_frames.json",
                "--batch-size",
                "5",
                "--provider",
                "auto",
                "--output",
                "tmp/annotations.json",
            ],
        ),
        StepSpec(
            name="compress",
            script="compress_and_store.py",
            args=[
                "--annotations",
                "tmp/annotations.json",
                "--frames-dir",
                "tmp/frames",
                "--output-dir",
                daily_dir,
                "--format",
                "webp",
                "--quality",
                "85",
                "--max-width",
                "1920",
            ],
        ),
        StepSpec(
            name="tags",
            script="build_tag_index.py",
            args=[
                "--index",
                f"{daily_dir}/_index.json",
                "--tag-index",
                "content-vault/tags/tag-index.json",
            ],
        ),
        StepSpec(
            name="log",
            script="update_log.py",
            args=[
                "--video-list",
                "tmp/video_list.json",
                "--annotations",
                "tmp/annotations.json",
                "--log",
                "content-vault/processing-log.json",
                "--date",
                pipeline_date,
            ],
        ),
        StepSpec(
            name="notion",
            script="notion_sync.py",
            args=["--annotations", "tmp/annotations.json"],
            optional=True,
        ),
        StepSpec(
            name="brief",
            script="generate_brief.py",
            args=[
                "--annotations",
                "tmp/annotations.json",
                "--transcripts-dir",
                "tmp/transcripts",
                "--watchlist",
                "config/watchlist.json",
                "--output",
                f"{daily_dir}/{brief_name}",
                "--provider",
                "auto",
            ],
            optional=True,
        ),
        StepSpec(
            name="archive",
            script="archive_transcripts.py",
            args=[
                "--transcripts-dir",
                "tmp/transcripts",
                "--output-dir",
                f"content-vault/transcripts/{pipeline_date}",
                "--annotations",
                "tmp/annotations.json",
                "--errors-dir",
                "content-vault/errors",
                "--date",
                pipeline_date,
            ],
        ),
        StepSpec(
            name="cleanup",
            script="cleanup_tmp.py",
            args=["--tmp-dir", "tmp"],
        ),
    ]


def create_single_video_list(video_url: str, output_path: Path, dry_run: bool) -> None:
    ensure_dir(output_path.parent)
    if dry_run:
        LOGGER.info("Dry-run: would inspect video URL %s", video_url)
        return

    cmd = ["yt-dlp", "--dump-single-json", "--no-playlist", video_url]
    response = run_command(cmd)
    payload = json.loads(response.stdout or "{}")

    video_id = str(payload.get("id") or "")
    if not video_id:
        raise RuntimeError(f"Unable to read video id from {video_url}")

    upload_raw = payload.get("upload_date")
    upload_date = parse_upload_date(str(upload_raw)) if upload_raw else None

    channel_name = str(payload.get("channel") or payload.get("uploader") or "Unknown Channel")
    channel_id = str(payload.get("channel_id") or "")
    channel_slug = normalize_slug(channel_name)

    entry = {
        "id": video_id,
        "url": video_url,
        "title": str(payload.get("title") or ""),
        "channelId": channel_id,
        "channelName": channel_name,
        "channelSlug": channel_slug,
        "uploadDate": upload_date.isoformat() if upload_date else None,
        "duration": int(payload.get("duration") or 0),
        "score": 1.0,
    }
    save_json(output_path, [entry])


def run_step(step: StepSpec, verbose: bool, dry_run: bool) -> tuple[int, str]:
    command = [sys.executable, script_path(step.script), *step.args]
    if verbose:
        command.append("--verbose")

    LOGGER.info("Step %s: %s", step.name, " ".join(command))

    if dry_run:
        print("DRY-RUN:", " ".join(command))
        return 0, "dry-run"

    completed = subprocess.run(command, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if stdout:
        LOGGER.info("%s stdout:\n%s", step.name, stdout)
    if stderr:
        LOGGER.info("%s stderr:\n%s", step.name, stderr)

    if completed.returncode != 0:
        return completed.returncode, stderr or stdout or "Step failed"

    return 0, stdout.splitlines()[-1] if stdout else "ok"


def mark_step(
    state: dict[str, Any],
    step_name: str,
    status: str,
    message: str = "",
) -> None:
    step_state = state.setdefault("steps", {}).setdefault(step_name, {})
    step_state["status"] = status
    step_state["message"] = message
    step_state["updatedAt"] = utcnow().isoformat()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    pipeline_date = normalize_date(args.date)
    # Generate a human-readable run_id for unique brief naming (multiple runs/day)
    run_id = utcnow().strftime("%Hh%M")
    steps = build_steps(pipeline_date, run_id=run_id)

    state_path = resolve_path("tmp/_pipeline_state.json")
    persist_state = not args.dry_run

    if persist_state:
        state = load_state(state_path, pipeline_date, args.video_url, args.force)
        # Keep state file unless cleanup step runs.
        save_state(state_path, state)
    else:
        state = default_state()
        state["date"] = pipeline_date
        state["videoUrl"] = args.video_url

    def persist() -> None:
        if persist_state:
            save_state(state_path, state)

    if args.video_url:
        select_step_state = state["steps"].get("select", {})
        if select_step_state.get("status") != "completed" or args.force:
            try:
                create_single_video_list(args.video_url, resolve_path("tmp/video_list.json"), args.dry_run)
                mark_step(state, "select", "completed", "single-video mode")
                persist()
                LOGGER.info("Single video mode enabled, select step skipped")
            except Exception as exc:  # noqa: BLE001
                mark_step(state, "select", "failed", str(exc))
                persist()
                raise

    fatal_error = False
    upstream_rerun = False  # Track if any upstream step was re-executed

    for step in steps:
        if args.video_url and step.name == "select":
            continue

        existing = state["steps"].get(step.name, {})
        status = existing.get("status")
        if status == "completed" and not args.force and not upstream_rerun:
            LOGGER.info("Skipping completed step: %s", step.name)
            continue

        if status == "completed" and upstream_rerun:
            LOGGER.info("Re-running %s because an upstream step was re-executed", step.name)

        mark_step(state, step.name, "running", "")
        persist()

        code, message = run_step(step, args.verbose, args.dry_run)
        if code == 0:
            mark_step(state, step.name, "completed", message)
            persist()
            upstream_rerun = True  # This step ran, so downstream steps need re-running
            continue

        if step.optional or step.name in {"notion", "transcribe"}:
            mark_step(state, step.name, "failed", message)
            persist()
            LOGGER.warning("Optional step %s failed; continuing", step.name)
            continue

        mark_step(state, step.name, "failed", message)
        persist()
        LOGGER.error("Step %s failed: %s", step.name, message)
        fatal_error = True
        break

    completed_steps = [
        name
        for name, payload in state.get("steps", {}).items()
        if isinstance(payload, dict) and payload.get("status") == "completed"
    ]
    failed_steps = [
        name
        for name, payload in state.get("steps", {}).items()
        if isinstance(payload, dict) and payload.get("status") == "failed"
    ]

    # Required steps (excludes optional: transcribe, notion, brief)
    REQUIRED_STEPS = {"select", "download", "extract", "window", "classify",
                      "compress", "tags", "log", "archive", "cleanup"}
    missing = sorted(REQUIRED_STEPS - set(completed_steps))
    all_required_done = len(missing) == 0

    summary = {
        "date": pipeline_date,
        "videoUrl": args.video_url,
        "dryRun": args.dry_run,
        "completed": completed_steps,
        "failed": failed_steps,
        "missing": missing,
        "verdict": "COMPLETE" if all_required_done else "INCOMPLETE",
        "stateFile": str(state_path) if persist_state else None,
    }

    print(json.dumps(summary, indent=2))

    if not all_required_done:
        LOGGER.error("Pipeline INCOMPLETE — missing required steps: %s", missing)

    return 1 if (fatal_error or not all_required_done) else 0


if __name__ == "__main__":
    raise SystemExit(main())
