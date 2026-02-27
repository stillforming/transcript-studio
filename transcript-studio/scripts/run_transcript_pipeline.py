#!/usr/bin/env python3
"""Run the Transcript Studio pipeline with resume support."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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

LOGGER = logging.getLogger("content_scout.run_transcript_pipeline")

DEFAULT_PRESET = "default"
VIDEO_LIST_PATH = "tmp/ts_video_list.json"
STATE_PATH = "tmp/_ts_pipeline_state.json"
TRANSCRIPT_LOG_PATH = "content-vault/transcript-studio-log.json"

STEP_ORDER = [
    "select",
    "download",
    "extract",
    "transcribe",
    "window",
    "classify",
    "merge",
    "summarize",
    "export",
    "log",
    "archive",
    "cleanup",
]


DEFAULT_STEP_TIMEOUT = 600  # 10 minutes


@dataclass(frozen=True)
class StepSpec:
    name: str
    script: str
    optional: bool = False
    per_video: bool = False
    timeout: int = DEFAULT_STEP_TIMEOUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Pipeline date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--video-url", help="Single YouTube URL to process")
    parser.add_argument(
        "--preset",
        choices=["default", "podcast", "presentation", "suno"],
        help="Optional preset override applied to all videos",
    )
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


def load_state(
    state_path: Path,
    pipeline_date: str,
    video_url: str | None,
    preset_override: str | None,
    force: bool,
) -> dict[str, Any]:
    if force or not state_path.exists():
        state = default_state()
        state["date"] = pipeline_date
        state["videoUrl"] = video_url
        state["presetOverride"] = preset_override
        return state

    state = load_json(state_path, default={})
    if not isinstance(state, dict):
        state = default_state()

    if (
        state.get("date") != pipeline_date
        or state.get("videoUrl") != video_url
        or state.get("presetOverride") != preset_override
    ):
        LOGGER.info("Existing state does not match request, creating a new state")
        state = default_state()

    state.setdefault("date", pipeline_date)
    state.setdefault("videoUrl", video_url)
    state.setdefault("presetOverride", preset_override)
    state.setdefault("steps", {step: {"status": "pending"} for step in STEP_ORDER})

    for step in STEP_ORDER:
        state["steps"].setdefault(step, {"status": "pending"})

    return state


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updatedAt"] = utcnow().isoformat()
    save_json(state_path, state)


def script_path(script_name: str) -> str:
    return str((Path(__file__).resolve().parent / script_name).resolve())


def build_steps() -> list[StepSpec]:
    return [
        StepSpec(name="select", script="select_transcript_videos.py"),
        StepSpec(name="download", script="download.py"),
        StepSpec(name="extract", script="extract_frames.py"),
        StepSpec(name="transcribe", script="transcribe_local.py"),
        StepSpec(name="window", script="window_transcripts.py"),
        StepSpec(name="classify", script="classify_annotate.py", timeout=1800),
        StepSpec(name="merge", script="merge_visuals.py", per_video=True),
        StepSpec(name="summarize", script="summarize_video.py", per_video=True, timeout=900),
        StepSpec(name="export", script="export_notion.py", per_video=True),
        StepSpec(name="log", script="update_log.py"),
        StepSpec(name="archive", script="archive_transcripts.py"),
        StepSpec(name="cleanup", script="cleanup_tmp.py"),
    ]


def load_preset(preset_name: str | None) -> dict[str, Any]:
    requested = str(preset_name or DEFAULT_PRESET).strip().lower() or DEFAULT_PRESET
    presets_dir = resolve_path("config/presets")
    default_path = presets_dir / "default.json"
    target_path = presets_dir / f"{requested}.json"

    selected_path = target_path
    if not target_path.exists():
        LOGGER.warning("Preset file not found for '%s', falling back to default", requested)
        selected_path = default_path

    payload = load_json(selected_path, default={})
    if not isinstance(payload, dict):
        payload = {}

    payload.setdefault("name", requested if selected_path == target_path else DEFAULT_PRESET)
    payload.setdefault("frame_interval", 15)
    payload.setdefault("hash_threshold", 5)
    payload.setdefault("whisper_model", "large-v3")
    return payload


def infer_pipeline_preset(video_list_path: Path) -> str:
    payload = load_json(video_list_path, default=[])
    if not isinstance(payload, list) or not payload:
        return DEFAULT_PRESET

    presets = {
        str(item.get("preset") or DEFAULT_PRESET).strip().lower() or DEFAULT_PRESET
        for item in payload
        if isinstance(item, dict)
    }
    presets.discard("")
    if not presets:
        return DEFAULT_PRESET
    if len(presets) == 1:
        return next(iter(presets))

    LOGGER.warning(
        "Multiple presets in selected videos (%s); using '%s' for global steps",
        ", ".join(sorted(presets)),
        DEFAULT_PRESET,
    )
    return DEFAULT_PRESET


def build_step_args(step_name: str, pipeline_date: str, preset: dict[str, Any]) -> list[str]:
    if step_name == "select":
        return [
            "--channels",
            "config/channels.json",
            "--playlists",
            "config/playlists.json",
            "--log",
            TRANSCRIPT_LOG_PATH,
            "--limit",
            "10",
            "--output",
            VIDEO_LIST_PATH,
        ]

    if step_name == "download":
        return [
            "--input",
            VIDEO_LIST_PATH,
            "--output-dir",
            "tmp/downloads",
            "--max-resolution",
            "720",
            "--delay",
            "5",
        ]

    if step_name == "extract":
        return [
            "--input-dir",
            "tmp/downloads",
            "--output-dir",
            "tmp/frames",
            "--interval",
            str(int(preset.get("frame_interval", 15))),
            "--hash-threshold",
            str(int(preset.get("hash_threshold", 5))),
        ]

    if step_name == "transcribe":
        return [
            "--input-dir",
            "tmp/downloads",
            "--output-dir",
            "tmp/transcripts",
            "--model",
            str(preset.get("whisper_model") or "large-v3"),
        ]

    if step_name == "window":
        return [
            "--frames-dir",
            "tmp/frames",
            "--transcripts-dir",
            "tmp/transcripts",
            "--window",
            "30",
            "--output",
            "tmp/windowed_frames.json",
        ]

    if step_name == "classify":
        return [
            "--input",
            "tmp/windowed_frames.json",
            "--batch-size",
            "5",
            "--provider",
            "auto",
            "--output",
            "tmp/annotations.json",
        ]

    if step_name == "log":
        return [
            "--video-list",
            VIDEO_LIST_PATH,
            "--annotations",
            "tmp/annotations.json",
            "--log",
            TRANSCRIPT_LOG_PATH,
            "--date",
            pipeline_date,
        ]

    if step_name == "archive":
        return [
            "--transcripts-dir",
            "tmp/transcripts",
            "--output-dir",
            f"content-vault/transcript-studio/transcripts/{pipeline_date}",
            "--annotations",
            "tmp/annotations.json",
            "--errors-dir",
            "content-vault/transcript-studio/errors",
            "--date",
            pipeline_date,
        ]

    if step_name == "cleanup":
        return ["--tmp-dir", "tmp"]

    raise ValueError(f"Unsupported step for static args: {step_name}")


def run_script(
    script: str,
    step_name: str,
    args: list[str],
    verbose: bool,
    dry_run: bool,
    timeout: int = DEFAULT_STEP_TIMEOUT,
) -> tuple[int, str]:
    command = [sys.executable, script_path(script), *args]
    if verbose:
        command.append("--verbose")

    LOGGER.info("Step %s: %s", step_name, " ".join(command))

    if dry_run:
        print("DRY-RUN:", " ".join(command))
        return 0, "dry-run"

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        LOGGER.error("Step %s timed out after %ss", step_name, timeout)
        return 124, f"Step timed out after {timeout}s"

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if stdout:
        LOGGER.info("%s stdout:\n%s", step_name, stdout)
    if stderr:
        LOGGER.info("%s stderr:\n%s", step_name, stderr)

    if completed.returncode != 0:
        return completed.returncode, stderr or stdout or "Step failed"
    return 0, stdout.splitlines()[-1] if stdout else "ok"


# ── Output validators ─────────────────────────────────────────────────────────
# Each returns an error message if the step's output is missing/invalid, or None if OK.


def _validate_download(workdir: Path) -> str | None:
    audio_files = list((workdir / "tmp" / "downloads").glob("*_audio.mp3"))
    if not audio_files:
        return "download produced no audio files in tmp/downloads/"
    return None


def _validate_extract(workdir: Path) -> str | None:
    frames_dir = workdir / "tmp" / "frames"
    if not frames_dir.exists():
        return "extract produced no tmp/frames/ directory"
    frame_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    if not frame_dirs:
        return "extract produced no frame directories in tmp/frames/"
    total_frames = sum(
        len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
        for d in frame_dirs
    )
    if total_frames == 0:
        return "extract produced 0 frames"
    return None


def _validate_transcribe(workdir: Path) -> str | None:
    transcripts = list((workdir / "tmp" / "transcripts").glob("*_transcript.json"))
    if not transcripts:
        return "transcribe produced no transcript files"
    for t in transcripts:
        payload = load_json(t, default={})
        segments = payload.get("segments", [])
        if segments and not payload.get("error"):
            return None
    return "all transcripts are empty or have errors"


def _validate_window(workdir: Path) -> str | None:
    wf = workdir / "tmp" / "windowed_frames.json"
    if not wf.exists():
        return "window produced no windowed_frames.json"
    data = load_json(wf, default=[])
    if not data:
        return "windowed_frames.json is empty"
    return None


def _validate_classify(workdir: Path) -> str | None:
    annotations = workdir / "tmp" / "annotations.json"
    if not annotations.exists():
        return "classify produced no annotations.json"
    data = load_json(annotations, default=[])
    if not data:
        return "annotations.json is empty"
    return None


def _validate_merge(workdir: Path) -> str | None:
    merged = list((workdir / "tmp").glob("ts_*_merged.json"))
    if not merged:
        return "merge produced no merged JSON files"
    return None


def _validate_summarize(workdir: Path) -> str | None:
    summaries = list((workdir / "tmp").glob("ts_*_summary.json"))
    if not summaries:
        return "summarize produced no summary JSON files"
    return None


STEP_VALIDATORS: dict[str, Callable[[Path], str | None]] = {
    "download": _validate_download,
    "extract": _validate_extract,
    "transcribe": _validate_transcribe,
    "window": _validate_window,
    "classify": _validate_classify,
    "merge": _validate_merge,
    "summarize": _validate_summarize,
}


def validate_step(step_name: str, workdir: Path) -> str | None:
    """Return error message if step output is invalid, None if OK."""
    validator = STEP_VALIDATORS.get(step_name)
    if validator is None:
        return None
    return validator(workdir)


def mark_step(state: dict[str, Any], step_name: str, status: str, message: str = "") -> None:
    step_state = state.setdefault("steps", {}).setdefault(step_name, {})
    step_state["status"] = status
    step_state["message"] = message
    step_state["updatedAt"] = utcnow().isoformat()


def load_video_entries(video_list_path: Path) -> list[dict[str, Any]]:
    payload = load_json(video_list_path, default=[])
    if not isinstance(payload, list):
        raise ValueError(f"Video list must be an array: {video_list_path}")
    return [item for item in payload if isinstance(item, dict)]


def video_id_of(entry: dict[str, Any]) -> str:
    return str(entry.get("id") or entry.get("videoId") or "").strip()


def create_single_video_list(
    video_url: str,
    output_path: Path,
    preset_override: str | None,
    dry_run: bool,
) -> None:
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
    preset = preset_override or DEFAULT_PRESET

    entry = {
        "id": video_id,
        "url": video_url,
        "title": str(payload.get("title") or ""),
        "channelId": channel_id,
        "channelName": channel_name,
        "channelSlug": channel_slug,
        "uploadDate": upload_date.isoformat() if upload_date else None,
        "duration": int(payload.get("duration") or 0),
        "source": "channel",
        "playlistName": None,
        "preset": preset,
    }
    save_json(output_path, [entry])


def ensure_window_transcript_aliases(transcripts_dir: Path, dry_run: bool) -> int:
    if not transcripts_dir.exists():
        return 0

    aliases = 0
    for transcript_path in sorted(transcripts_dir.glob("*_transcript.json")):
        video_id = transcript_path.stem.replace("_transcript", "")
        if not video_id:
            continue
        alias_path = transcripts_dir / f"{video_id}.json"

        if dry_run:
            aliases += 1
            continue

        shutil.copy2(transcript_path, alias_path)
        aliases += 1

    return aliases


def merge_transcript_metadata(
    transcript_path: Path,
    video_entry: dict[str, Any],
    preset_override: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    if not transcript_path.exists():
        LOGGER.warning("Transcript file missing for metadata merge: %s", transcript_path)
        return

    payload = load_json(transcript_path, default={})
    if not isinstance(payload, dict):
        LOGGER.warning("Unexpected transcript payload shape: %s", transcript_path)
        return

    effective_preset = str(preset_override or video_entry.get("preset") or DEFAULT_PRESET).strip() or DEFAULT_PRESET
    payload["preset"] = effective_preset

    playlist_name = str(video_entry.get("playlistName") or "").strip()
    if playlist_name:
        payload["playlist"] = playlist_name

    video_url = str(video_entry.get("url") or "").strip()
    if video_url and not str(payload.get("url") or "").strip():
        payload["url"] = video_url

    video_title = str(video_entry.get("title") or "").strip()
    if video_title and not str(payload.get("videoTitle") or "").strip():
        payload["videoTitle"] = video_title

    channel_name = str(video_entry.get("channelName") or "").strip()
    if channel_name and not str(payload.get("channelName") or "").strip():
        payload["channelName"] = channel_name

    channel_slug = str(video_entry.get("channelSlug") or "").strip()
    if channel_slug and not str(payload.get("channelSlug") or "").strip():
        payload["channelSlug"] = channel_slug

    save_json(transcript_path, payload)


def run_per_video_step(
    step: StepSpec,
    videos: list[dict[str, Any]],
    preset_override: str | None,
    verbose: bool,
    dry_run: bool,
) -> tuple[int, str]:
    if not videos:
        return 0, "no videos selected"

    failures = 0
    processed = 0
    last_message = "ok"

    for video in videos:
        video_id = video_id_of(video)
        if not video_id:
            failures += 1
            continue

        transcript_path = f"tmp/transcripts/{video_id}_transcript.json"
        merge_transcript_metadata(resolve_path(transcript_path), video, preset_override, dry_run)

        if step.name == "merge":
            args = [
                "--transcript",
                transcript_path,
                "--annotations",
                "tmp/annotations.json",
                "--frames-dir",
                "tmp/frames",
                "--output",
                f"tmp/ts_{video_id}_merged.json",
            ]
        elif step.name == "summarize":
            args = [
                "--transcript",
                transcript_path,
                "--annotations",
                "tmp/annotations.json",
                "--output",
                f"tmp/ts_{video_id}_summary.json",
            ]
        elif step.name == "export":
            args = [
                "--transcript",
                transcript_path,
                "--merged",
                f"tmp/ts_{video_id}_merged.json",
                "--summary",
                f"tmp/ts_{video_id}_summary.json",
            ]
        else:
            raise ValueError(f"Unsupported per-video step: {step.name}")

        code, message = run_script(step.script, f"{step.name}:{video_id}", args, verbose, dry_run)
        if code != 0:
            failures += 1
            last_message = message
            continue

        processed += 1
        last_message = message

    if failures:
        return 1, f"{step.name} failed for {failures} video(s): {last_message}"
    return 0, f"{step.name} completed for {processed} video(s)"


def cleanup_transcript_tmp_artifacts(tmp_dir: Path, dry_run: bool) -> int:
    targets: list[Path] = [tmp_dir / "ts_video_list.json"]
    targets.extend(sorted(tmp_dir.glob("ts_*_merged.json")))
    targets.extend(sorted(tmp_dir.glob("ts_*_summary.json")))

    removed = 0
    for target in targets:
        if not target.exists():
            continue
        if dry_run:
            removed += 1
            continue
        target.unlink(missing_ok=True)
        removed += 1
    return removed


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    pipeline_date = normalize_date(args.date)
    steps = build_steps()

    state_path = resolve_path(STATE_PATH)
    persist_state = not args.dry_run

    if persist_state:
        state = load_state(state_path, pipeline_date, args.video_url, args.preset, args.force)
        save_state(state_path, state)
    else:
        state = default_state()
        state["date"] = pipeline_date
        state["videoUrl"] = args.video_url
        state["presetOverride"] = args.preset

    def persist() -> None:
        if persist_state:
            save_state(state_path, state)

    if args.video_url:
        select_step_state = state["steps"].get("select", {})
        if select_step_state.get("status") != "completed" or args.force:
            try:
                create_single_video_list(
                    args.video_url,
                    resolve_path(VIDEO_LIST_PATH),
                    args.preset,
                    args.dry_run,
                )
                mark_step(state, "select", "completed", "single-video mode")
                persist()
                LOGGER.info("Single video mode enabled, select step skipped")
            except Exception as exc:  # noqa: BLE001
                mark_step(state, "select", "failed", str(exc))
                persist()
                raise

    active_preset_name = args.preset or DEFAULT_PRESET
    active_preset = load_preset(active_preset_name)

    fatal_error = False
    upstream_rerun = False

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

        if step.name in {"extract", "transcribe"} and not args.preset:
            inferred_name = infer_pipeline_preset(resolve_path(VIDEO_LIST_PATH))
            if inferred_name != active_preset_name:
                active_preset_name = inferred_name
                active_preset = load_preset(active_preset_name)
            LOGGER.info(
                "Using preset '%s' (interval=%s hash_threshold=%s whisper_model=%s)",
                active_preset.get("name", active_preset_name),
                active_preset.get("frame_interval"),
                active_preset.get("hash_threshold"),
                active_preset.get("whisper_model"),
            )

        mark_step(state, step.name, "running", "")
        persist()

        if step.name == "window":
            alias_count = ensure_window_transcript_aliases(resolve_path("tmp/transcripts"), args.dry_run)
            if alias_count:
                LOGGER.info("Prepared %s transcript alias file(s) for window step", alias_count)

        if step.per_video:
            videos = load_video_entries(resolve_path(VIDEO_LIST_PATH))
            if step.name == "export" and not os.environ.get("NOTION_TOKEN"):
                code, message = 0, "skipped export (NOTION_TOKEN not set)"
            else:
                code, message = run_per_video_step(step, videos, args.preset, args.verbose, args.dry_run)
        else:
            step_args = build_step_args(step.name, pipeline_date, active_preset)
            code, message = run_script(
                step.script, step.name, step_args, args.verbose, args.dry_run,
                timeout=step.timeout,
            )

            if code == 0 and step.name == "cleanup":
                removed = cleanup_transcript_tmp_artifacts(
                    resolve_path("tmp"),
                    args.dry_run,
                )
                if removed:
                    message = f"{message}; removed {removed} transcript-studio temp files"

        if code == 0:
            validation_error = validate_step(step.name, resolve_path("."))
            if validation_error:
                LOGGER.error(
                    "Step %s passed (exit 0) but output validation failed: %s",
                    step.name, validation_error,
                )
                mark_step(state, step.name, "failed", f"output validation: {validation_error}")
                persist()
                fatal_error = True
                break
            mark_step(state, step.name, "completed", message)
            persist()
            upstream_rerun = True
            continue

        if step.optional:
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

    REQUIRED_STEPS = {"select", "download", "extract", "transcribe", "window",
                      "classify", "merge", "summarize", "export"}
    missing = sorted(REQUIRED_STEPS - set(completed_steps))
    all_required_done = len(missing) == 0

    summary = {
        "date": pipeline_date,
        "videoUrl": args.video_url,
        "presetOverride": args.preset,
        "activePreset": active_preset.get("name", active_preset_name),
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
