#!/usr/bin/env python3
"""Transcribe downloaded audio files with local mlx-whisper and OpenAI fallback."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from _common import ensure_dir, load_json, resolve_path, run_command, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.transcribe_local")

DEFAULT_MLX_MODEL = "large-v3"
DEFAULT_OPENAI_MODEL = "whisper-1"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

MLX_MODEL_ALIASES: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

_DIARIZATION_PIPELINE: Any | None = None
_DIARIZATION_DISABLED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="tmp/downloads", help="Directory containing downloaded assets")
    parser.add_argument("--output-dir", default="tmp/transcripts", help="Directory to write transcript files")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name (default: {DEFAULT_MLX_MODEL} for mlx, {DEFAULT_OPENAI_MODEL} for OpenAI)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "mlx", "openai"],
        default="auto",
        help="Transcription engine selection",
    )
    diarize_group = parser.add_mutually_exclusive_group()
    diarize_group.add_argument("--diarize", dest="diarize", action="store_true", help="Enable speaker diarization")
    diarize_group.add_argument("--no-diarize", dest="diarize", action="store_false", help="Disable diarization")
    parser.set_defaults(diarize=True)
    parser.add_argument("--max-file-size-mb", type=int, default=25, help="Split files above this size for OpenAI")
    parser.add_argument("--chunk-duration", type=int, default=600, help="Chunk duration in seconds for OpenAI")
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
    if hasattr(response, "__dict__"):
        as_dict = dict(getattr(response, "__dict__"))
        if as_dict:
            return as_dict
    if isinstance(response, str):
        return json.loads(response)
    return json.loads(str(response))


def normalize_segment(segment: Any) -> dict[str, Any] | None:
    payload: dict[str, Any]
    if isinstance(segment, dict):
        payload = segment
    else:
        try:
            payload = to_dict(segment)
        except Exception:  # noqa: BLE001
            return None

    try:
        start = float(payload.get("start", 0.0))
        end = float(payload.get("end", start))
    except (TypeError, ValueError):
        return None

    text = str(payload.get("text") or "").strip()
    if not text:
        return None

    return {
        "start": max(0.0, start),
        "end": max(max(0.0, start), end),
        "text": text,
    }


def normalize_segments(raw_segments: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return []
    normalized: list[dict[str, Any]] = []
    for segment in raw_segments:
        clean = normalize_segment(segment)
        if clean is not None:
            normalized.append(clean)
    return normalized


def normalize_mlx_model(model: str) -> str:
    raw = model.strip()
    key = raw.lower()
    if "/" in raw:
        return raw
    return MLX_MODEL_ALIASES.get(key, raw)


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def minute_mark_from_start(seconds: float) -> str:
    minute_floor = (max(0, int(seconds)) // 60) * 60
    return format_timestamp(minute_floor)


def detect_engine(requested_engine: str) -> str:
    if requested_engine == "mlx":
        try:
            import mlx_whisper  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("mlx engine requested but mlx_whisper is unavailable") from exc
        return "mlx"

    if requested_engine == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when --engine=openai")
        return "openai"

    try:
        import mlx_whisper  # noqa: F401

        return "mlx"
    except Exception:  # noqa: BLE001
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        raise RuntimeError(
            "Could not auto-detect transcription engine. Install mlx-whisper "
            "or set OPENAI_API_KEY for OpenAI fallback."
        ) from None


def resolve_model(engine: str, model_arg: str | None) -> str:
    if model_arg and model_arg.strip():
        model = model_arg.strip()
        # Remap mlx model names to OpenAI when falling back
        if engine == "openai" and model in MLX_MODEL_ALIASES:
            LOGGER.info("Remapping mlx model '%s' to '%s' for OpenAI engine", model, DEFAULT_OPENAI_MODEL)
            return DEFAULT_OPENAI_MODEL
        return model
    if engine == "mlx":
        return DEFAULT_MLX_MODEL
    return DEFAULT_OPENAI_MODEL


def transcribe_mlx(audio_path: Path, model: str) -> list[dict[str, Any]]:
    import mlx_whisper

    resolved_model = normalize_mlx_model(model)
    LOGGER.debug("mlx-whisper model resolved to %s", resolved_model)

    result: Any
    errors: list[Exception] = []

    for call in (
        lambda: mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=resolved_model),
        lambda: mlx_whisper.transcribe(str(audio_path), model=resolved_model),
        lambda: mlx_whisper.transcribe(str(audio_path), resolved_model),
    ):
        try:
            result = call()
            break
        except TypeError as exc:
            errors.append(exc)
    else:
        if errors:
            raise errors[-1]
        result = mlx_whisper.transcribe(str(audio_path))

    payload = to_dict(result)
    segments = normalize_segments(payload.get("segments", []))
    if segments:
        return segments

    text = str(payload.get("text") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": 0.0, "text": text}]


def build_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def transcribe_openai_chunk(client: Any, file_path: Path, model: str) -> dict[str, Any]:
    with file_path.open("rb") as audio_handle:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_handle,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    payload = to_dict(response)
    payload["segments"] = normalize_segments(payload.get("segments", []))
    payload["text"] = str(payload.get("text") or "").strip()
    return payload


def transcribe_openai(
    client: Any,
    audio_path: Path,
    model: str,
    max_file_size_mb: int,
    chunk_duration: int,
    chunks_root: Path,
) -> list[dict[str, Any]]:
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    chunk_payloads: list[dict[str, Any]] = []

    if size_mb > max_file_size_mb:
        chunks_dir = ensure_dir(chunks_root / audio_path.stem)
        chunks = split_audio(audio_path, chunks_dir, chunk_duration)
        if not chunks:
            raise RuntimeError("Audio split produced no chunks")
        try:
            for chunk in chunks:
                chunk_payloads.append(transcribe_openai_chunk(client, chunk, model))
        finally:
            shutil.rmtree(chunks_dir, ignore_errors=True)
    else:
        chunk_payloads.append(transcribe_openai_chunk(client, audio_path, model))

    merged_segments: list[dict[str, Any]] = []
    for index, chunk_payload in enumerate(chunk_payloads):
        offset = index * chunk_duration
        for segment in chunk_payload.get("segments", []):
            try:
                start = float(segment.get("start", 0.0)) + offset
                end = float(segment.get("end", start)) + offset
            except (TypeError, ValueError):
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            merged_segments.append(
                {
                    "start": max(0.0, start),
                    "end": max(max(0.0, start), end),
                    "text": text,
                }
            )
    return merged_segments


def load_diarization_pipeline() -> Any | None:
    global _DIARIZATION_PIPELINE, _DIARIZATION_DISABLED

    if _DIARIZATION_DISABLED:
        return None
    if _DIARIZATION_PIPELINE is not None:
        return _DIARIZATION_PIPELINE

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if not hf_token:
        LOGGER.warning("HF_TOKEN not set; speaker diarization disabled.")
        _DIARIZATION_DISABLED = True
        return None

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("pyannote.audio not available (%s); speaker diarization disabled.", exc)
        _DIARIZATION_DISABLED = True
        return None

    try:
        _DIARIZATION_PIPELINE = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=hf_token)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to initialize pyannote diarization (%s); disabled.", exc)
        _DIARIZATION_DISABLED = True
        return None

    return _DIARIZATION_PIPELINE


def diarize(audio_path: Path) -> list[tuple[float, float, str]]:
    pipeline = load_diarization_pipeline()
    if pipeline is None:
        return []

    try:
        diarization = pipeline(str(audio_path))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Speaker diarization failed for %s: %s", audio_path.name, exc)
        return []

    speaker_labels: dict[str, str] = {}
    segments: list[tuple[float, float, str]] = []
    for turn, _, raw_speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end <= start:
            continue

        key = str(raw_speaker)
        if key not in speaker_labels:
            speaker_labels[key] = f"Speaker {len(speaker_labels) + 1}"
        segments.append((start, end, speaker_labels[key]))

    segments.sort(key=lambda item: item[0])
    return segments


def best_speaker_for_segment(
    start: float,
    end: float,
    diarization_segments: list[tuple[float, float, str]],
) -> str:
    if not diarization_segments:
        return "Speaker 1"

    best_overlap = 0.0
    best_label: str | None = None

    for dia_start, dia_end, dia_speaker in diarization_segments:
        overlap = min(end, dia_end) - max(start, dia_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = dia_speaker

    if best_label is not None and best_overlap > 0:
        return best_label

    midpoint = (start + end) / 2.0
    nearest = min(diarization_segments, key=lambda item: abs(((item[0] + item[1]) / 2.0) - midpoint))
    return nearest[2]


def merge_diarization(
    whisper_segments: list[dict[str, Any]],
    diarization_segments: list[tuple[float, float, str]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in whisper_segments:
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue

        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        speaker = best_speaker_for_segment(start, end, diarization_segments)
        merged.append(
            {
                "start": max(0.0, start),
                "end": max(max(0.0, start), end),
                "text": text,
                "speaker": speaker,
                "minute_mark": minute_mark_from_start(start),
            }
        )
    return merged


def format_raw_transcript(segments: list[dict[str, Any]]) -> str:
    chunks = [str(segment.get("text") or "").strip() for segment in segments]
    return " ".join(chunk for chunk in chunks if chunk).strip()


def format_clean_transcript(segments: list[dict[str, Any]]) -> str:
    if not segments:
        return ""

    ordered = sorted(segments, key=lambda segment: float(segment.get("start", 0.0)))
    lines: list[str] = []
    current_speaker: str | None = None
    paragraph_parts: list[str] = []
    next_marker_minute = 0

    def flush_paragraph() -> None:
        nonlocal current_speaker, paragraph_parts
        if current_speaker is None or not paragraph_parts:
            return
        lines.append(f"{current_speaker}: {' '.join(paragraph_parts).strip()}")
        lines.append("")
        current_speaker = None
        paragraph_parts = []

    for segment in ordered:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        start = max(0, int(float(segment.get("start", 0.0))))
        target_minute = start // 60

        while next_marker_minute <= target_minute:
            flush_paragraph()
            marker = format_timestamp(next_marker_minute * 60)
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"--- [{marker}] ---")
            lines.append("")
            next_marker_minute += 1

        speaker = str(segment.get("speaker") or "Speaker 1")
        if current_speaker == speaker:
            paragraph_parts.append(text)
        else:
            flush_paragraph()
            current_speaker = speaker
            paragraph_parts = [text]

    flush_paragraph()
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def collect_speakers(segments: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    speakers: list[str] = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "Speaker 1")
        if speaker not in seen:
            seen.add(speaker)
            speakers.append(speaker)
    return speakers


def build_transcript_payload(
    metadata: dict[str, Any],
    video_id: str,
    segments: list[dict[str, Any]],
    engine: str,
    model: str,
    diarization_enabled: bool,
    error: str | None = None,
) -> dict[str, Any]:
    raw_text = format_raw_transcript(segments)
    full_text = format_clean_transcript(segments)
    payload: dict[str, Any] = {
        "videoId": video_id,
        "videoTitle": metadata.get("title", ""),
        "channelName": metadata.get("channelName", ""),
        "channelSlug": metadata.get("channelSlug", ""),
        "url": metadata.get("url"),
        "engine": engine,
        "model": model,
        "diarizationEnabled": diarization_enabled,
        "segments": segments,
        "speakers": collect_speakers(segments) or ["Speaker 1"],
        "full_text": full_text,
        "raw_text": raw_text,
        "text": raw_text,
    }
    if error:
        payload["error"] = error
    return payload


def write_outputs(output_dir: Path, video_id: str, payload: dict[str, Any]) -> None:
    transcript_path = output_dir / f"{video_id}_transcript.json"
    clean_path = output_dir / f"{video_id}_clean.txt"
    raw_path = output_dir / f"{video_id}_raw.txt"

    save_json(transcript_path, payload)
    clean_text = str(payload.get("full_text") or "")
    raw_text = str(payload.get("raw_text") or "")
    clean_path.write_text(clean_text, encoding="utf-8")
    raw_path.write_text(raw_text, encoding="utf-8")


def process_video(
    meta_path: Path,
    input_dir: Path,
    output_dir: Path,
    engine: str,
    model: str,
    diarization_enabled: bool,
    openai_client: Any | None,
    max_file_size_mb: int,
    chunk_duration: int,
) -> bool:
    metadata = load_json(meta_path, default={})
    video_id = str(metadata.get("videoId") or meta_path.stem.replace("_meta", ""))
    audio_file = resolve_audio_file(input_dir, video_id, metadata)

    if audio_file is None:
        payload = build_transcript_payload(
            metadata=metadata,
            video_id=video_id,
            segments=[],
            engine=engine,
            model=model,
            diarization_enabled=False,
            error="Audio file not found",
        )
        write_outputs(output_dir, video_id, payload)
        LOGGER.warning("Skipping %s because audio file is missing", video_id)
        return False

    if engine == "mlx":
        whisper_segments = transcribe_mlx(audio_file, model)
    else:
        if openai_client is None:
            raise RuntimeError("OpenAI client is not initialized")
        whisper_segments = transcribe_openai(
            client=openai_client,
            audio_path=audio_file,
            model=model,
            max_file_size_mb=max_file_size_mb,
            chunk_duration=chunk_duration,
            chunks_root=output_dir / "_chunks" / video_id,
        )

    diarization_segments: list[tuple[float, float, str]] = []
    diarization_used = False
    if diarization_enabled:
        diarization_segments = diarize(audio_file)
        diarization_used = bool(diarization_segments)
        if not diarization_used:
            LOGGER.info("Diarization unavailable for %s; using single-speaker fallback.", video_id)

    segments = merge_diarization(whisper_segments, diarization_segments)
    payload = build_transcript_payload(
        metadata=metadata,
        video_id=video_id,
        segments=segments,
        engine=engine,
        model=model,
        diarization_enabled=diarization_enabled and diarization_used,
    )
    write_outputs(output_dir, video_id, payload)
    return True


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    input_dir = resolve_path(args.input_dir)
    output_dir = ensure_dir(resolve_path(args.output_dir))

    engine = detect_engine(args.engine)
    model = resolve_model(engine, args.model)
    LOGGER.info("Using transcription engine=%s model=%s", engine, model)

    openai_client: Any | None = None
    if engine == "openai":
        openai_client = build_openai_client()

    metas = iter_meta_files(input_dir)
    if not metas:
        LOGGER.warning("No metadata files found in %s", input_dir)

    transcribed = 0
    failed = 0

    for meta_path in metas:
        try:
            ok = process_video(
                meta_path=meta_path,
                input_dir=input_dir,
                output_dir=output_dir,
                engine=engine,
                model=model,
                diarization_enabled=args.diarize,
                openai_client=openai_client,
                max_file_size_mb=args.max_file_size_mb,
                chunk_duration=args.chunk_duration,
            )
            if ok:
                transcribed += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            metadata = load_json(meta_path, default={})
            video_id = str(metadata.get("videoId") or meta_path.stem.replace("_meta", ""))
            payload = build_transcript_payload(
                metadata=metadata,
                video_id=video_id,
                segments=[],
                engine=engine,
                model=model,
                diarization_enabled=False,
                error=f"Transcription failed: {exc}",
            )
            write_outputs(output_dir, video_id, payload)
            LOGGER.exception("Transcription failed for %s: %s", video_id, exc)

    LOGGER.info("Transcription complete (success=%s failed=%s)", transcribed, failed)
    print(f"Transcribed {transcribed} videos (failed {failed})")
    if transcribed == 0 and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
