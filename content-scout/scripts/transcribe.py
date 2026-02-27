#!/usr/bin/env python3
"""Transcribe downloaded audio files using OpenAI Whisper."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from openai import OpenAI

from _common import ensure_dir, load_json, resolve_path, run_command, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.transcribe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="tmp/downloads", help="Directory containing downloaded assets")
    parser.add_argument("--output-dir", default="tmp/transcripts", help="Directory to write transcript JSON files")
    parser.add_argument("--model", default="whisper-1", help="Whisper model name")
    parser.add_argument("--max-file-size-mb", type=int, default=25, help="Split files above this size")
    parser.add_argument("--chunk-duration", type=int, default=600, help="Chunk duration in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def iter_meta_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_meta.json"))


def resolve_audio_file(input_dir: Path, video_id: str, metadata: dict[str, Any]) -> Path | None:
    audio_name = metadata.get("audioFile")
    if isinstance(audio_name, str) and audio_name:
        candidate = input_dir / audio_name
        if candidate.exists():
            return candidate

    candidate = input_dir / f"{video_id}_audio.mp3"
    if candidate.exists():
        return candidate
    return None


def split_audio(input_file: Path, chunks_dir: Path, chunk_duration: int) -> list[Path]:
    ensure_dir(chunks_dir)
    pattern = chunks_dir / "chunk_%03d.mp3"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration),
        "-c",
        "copy",
        str(pattern),
    ]
    LOGGER.debug("Running: %s", " ".join(cmd))
    run_command(cmd)
    return sorted(chunks_dir.glob("chunk_*.mp3"))


def to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[no-any-return]
    if hasattr(response, "to_dict"):
        return response.to_dict()  # type: ignore[no-any-return]
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return json.loads(response)
    return json.loads(str(response))


def transcribe_chunk(client: OpenAI, file_path: Path, model: str) -> dict[str, Any]:
    with file_path.open("rb") as audio_handle:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_handle,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    payload = to_dict(response)
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        payload["segments"] = []
    return payload


def merge_transcripts(chunks: list[dict[str, Any]], chunk_duration: int) -> dict[str, Any]:
    merged_segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    language = None

    for index, chunk_payload in enumerate(chunks):
        offset = index * chunk_duration
        language = language or chunk_payload.get("language")
        text = str(chunk_payload.get("text") or "").strip()
        if text:
            text_parts.append(text)

        for segment in chunk_payload.get("segments", []):
            try:
                start = float(segment.get("start", 0.0)) + offset
                end = float(segment.get("end", start)) + offset
            except (TypeError, ValueError):
                continue
            merged_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": str(segment.get("text") or "").strip(),
                }
            )

    return {
        "language": language,
        "text": " ".join(text_parts).strip(),
        "segments": merged_segments,
    }


def build_empty_transcript(metadata: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "videoId": metadata.get("videoId"),
        "videoTitle": metadata.get("title", ""),
        "channelName": metadata.get("channelName", ""),
        "channelSlug": metadata.get("channelSlug", ""),
        "url": metadata.get("url"),
        "language": None,
        "text": "",
        "segments": [],
        "error": reason,
    }


def process_video(
    client: OpenAI,
    meta_path: Path,
    input_dir: Path,
    output_dir: Path,
    model: str,
    max_file_size_mb: int,
    chunk_duration: int,
) -> bool:
    metadata = load_json(meta_path, default={})
    video_id = str(metadata.get("videoId") or meta_path.stem.replace("_meta", ""))
    audio_file = resolve_audio_file(input_dir, video_id, metadata)

    if audio_file is None:
        payload = build_empty_transcript(metadata, "Audio file not found")
        payload["videoId"] = video_id
        save_json(output_dir / f"{video_id}.json", payload)
        LOGGER.warning("Skipping %s because audio file is missing", video_id)
        return False

    size_mb = audio_file.stat().st_size / (1024 * 1024)
    chunk_payloads: list[dict[str, Any]] = []

    if size_mb > max_file_size_mb:
        chunks_dir = ensure_dir(output_dir / "_chunks" / video_id)
        chunks = split_audio(audio_file, chunks_dir, chunk_duration)
        if not chunks:
            payload = build_empty_transcript(metadata, "Audio split produced no chunks")
            payload["videoId"] = video_id
            save_json(output_dir / f"{video_id}.json", payload)
            return False
        for chunk in chunks:
            chunk_payloads.append(transcribe_chunk(client, chunk, model))
        shutil.rmtree(chunks_dir, ignore_errors=True)
    else:
        chunk_payloads.append(transcribe_chunk(client, audio_file, model))

    merged = merge_transcripts(chunk_payloads, chunk_duration)
    payload = {
        "videoId": video_id,
        "videoTitle": metadata.get("title", ""),
        "channelName": metadata.get("channelName", ""),
        "channelSlug": metadata.get("channelSlug", ""),
        "url": metadata.get("url"),
        "language": merged.get("language"),
        "text": merged.get("text", ""),
        "segments": merged.get("segments", []),
    }
    save_json(output_dir / f"{video_id}.json", payload)
    return True


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    input_dir = resolve_path(args.input_dir)
    output_dir = ensure_dir(resolve_path(args.output_dir))

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for transcription")

    client = OpenAI()
    metas = iter_meta_files(input_dir)

    transcribed = 0
    failed = 0

    for meta_path in metas:
        try:
            ok = process_video(
                client,
                meta_path,
                input_dir,
                output_dir,
                args.model,
                args.max_file_size_mb,
                args.chunk_duration,
            )
            if ok:
                transcribed += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            metadata = load_json(meta_path, default={})
            video_id = str(metadata.get("videoId") or meta_path.stem.replace("_meta", ""))
            save_json(
                output_dir / f"{video_id}.json",
                build_empty_transcript(metadata, f"Transcription failed: {exc}"),
            )
            LOGGER.exception("Transcription failed for %s: %s", video_id, exc)

    LOGGER.info("Transcription complete (success=%s failed=%s)", transcribed, failed)
    print(f"Transcribed {transcribed} videos (failed {failed})")
    if transcribed == 0 and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
