#!/usr/bin/env python3
"""Classify and annotate frames using vision models (Anthropic or OpenAI)."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from _common import ROOT_DIR, load_json, resolve_path, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.classify_annotate")

PROMPT_TEMPLATE = """You are a YouTube content strategist analyzing a competitor's video for an options trader who runs his own channel.
Video: "{title}" by {channel}

Your goal: Extract CONTENT INTELLIGENCE — what makes this video work (or not) as content? How is the creator structuring their argument? What can our client learn or respond to?

For EACH frame, classify and annotate.

Categories: CHART_VISUAL, TALKING_HEAD, GRAPHIC, SCREEN, TABLE, SLIDE, FILLER
- TALKING_HEAD: presenter on camera. Note energy, style, whether they're explaining or reacting.
- CHART_VISUAL: a chart being used to support a point. Focus on HOW they present it, not the price data.
- GRAPHIC: custom overlays, titles, lower-thirds, comparisons.
- SCREEN: screen recording of a trading platform or website.
- TABLE/SLIDE: data tables, presentation slides.
- FILLER: intros, outros, sponsor reads, dead air.

If FILLER, return category + confidence only.

For all other categories, extract:
- what: describe what's shown and how it's being used in the video's narrative
- verbal_context: quote the most important sentence from transcriptWindow — what point is the presenter making HERE?
- content_format: what type of content moment is this? (education, prediction, trade-recap, reaction, tutorial, storytelling, call-to-action, opinion)
- visual_technique: how are they presenting this visually? (clean chart, annotated chart, side-by-side comparison, picture-in-picture, full-screen graphic, talking-head-with-overlay, etc.)
- topic: the subject being discussed (e.g. "NVDA earnings positioning", "SPY crash scenario", "iron condor strategy")
- insight: what's the interesting or novel point being made? What might our client learn from this?
- hook_quality: 1-5 (5 = compelling moment that keeps viewers watching, 1 = boring/repetitive)
- content_idea: a specific video idea our client could make in response to or inspired by this moment
- tags: array of topic tags (ticker names, strategy names, concepts)
- ticker: string or null

Respond as JSON array only. One object per input frame.
"""

ALLOWED_CATEGORIES = {"CHART_VISUAL", "TALKING_HEAD", "GRAPHIC", "SCREEN", "TABLE", "SLIDE"}
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-5.2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tmp/windowed_frames.json", help="Windowed frames JSON path")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of frames per API call")
    parser.add_argument("--provider", choices=["anthropic", "openai", "auto"], default="auto",
                        help="LLM provider (default: auto-detect from available API keys)")
    parser.add_argument("--model", default=None, help="Model name (default: provider-specific)")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Minimum confidence required for kept annotations")
    parser.add_argument("--output", default="tmp/annotations.json", help="Output annotations JSON path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def detect_provider(requested: str) -> str:
    """Detect which provider to use based on available API keys."""
    if requested != "auto":
        return requested

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if has_anthropic:
        return "anthropic"
    if has_openai:
        return "openai"

    raise RuntimeError(
        "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment."
    )


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def resolve_frame_path(frame_path: str) -> Path:
    candidate = Path(frame_path)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_json_array(raw_text: str) -> list[dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    fence_match = JSON_BLOCK_RE.search(text)
    if fence_match:
        try:
            payload = json.loads(fence_match.group(1).strip())
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    LOGGER.warning("Could not parse model response as JSON array")
    return []


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def normalize_category(value: Any) -> str:
    category = str(value or "FILLER").strip().upper()
    if not category:
        category = "FILLER"
    return category


def normalize_annotation(frame: dict[str, Any], model_payload: dict[str, Any], threshold: float) -> dict[str, Any]:
    category = normalize_category(model_payload.get("category"))
    try:
        confidence = float(model_payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    hook_quality_value = model_payload.get("hook_quality")
    try:
        hook_quality = int(hook_quality_value) if hook_quality_value is not None else None
    except (TypeError, ValueError):
        hook_quality = None

    kept = category in ALLOWED_CATEGORIES and confidence >= threshold

    return {
        "videoId": frame.get("videoId"),
        "videoTitle": frame.get("videoTitle"),
        "channelName": frame.get("channelName"),
        "channelSlug": frame.get("channelSlug"),
        "framePath": frame.get("framePath"),
        "timestamp": frame.get("timestamp"),
        "sourceUrl": frame.get("sourceUrl"),
        "transcriptWindow": frame.get("transcriptWindow", ""),
        "category": category,
        "confidence": confidence,
        "kept": kept,
        "description": model_payload.get("what") or model_payload.get("description"),
        "verbal_context": model_payload.get("verbal_context"),
        "content_format": model_payload.get("content_format"),
        "visual_technique": model_payload.get("visual_technique"),
        "topic": model_payload.get("topic"),
        "insight": model_payload.get("insight"),
        "hook_quality": hook_quality,
        "content_idea": model_payload.get("content_idea"),
        "tags": normalize_tags(model_payload.get("tags")),
        "ticker": model_payload.get("ticker"),
        "raw": model_payload,
    }


def build_frame_text(batch: list[dict[str, Any]]) -> str:
    """Build the per-frame text descriptions (shared between providers)."""
    parts: list[str] = []
    for idx, frame in enumerate(batch, start=1):
        parts.append(
            f"Frame {idx}:\n"
            f"framePath: {frame.get('framePath', '')}\n"
            f"videoId: {frame.get('videoId', '')}\n"
            f"timestamp: {frame.get('timestamp', '')}\n"
            f"sourceUrl: {frame.get('sourceUrl', '')}\n"
            f"transcriptWindow: {frame.get('transcriptWindow', '')}\n"
        )
    return "\n".join(parts)


RESPONSE_INSTRUCTION = (
    "Return a JSON array only. One object per input frame. "
    "Each object MUST include: framePath, category, confidence, what, verbal_context, "
    "content_format, visual_technique, topic, insight, hook_quality, content_idea, tags, ticker."
)


# ─── Anthropic provider ───────────────────────────────────────────────────

def call_anthropic(batch: list[dict[str, Any]], model: str, prompt: str) -> str:
    """Send a batch to Anthropic's vision API and return raw text response."""
    from anthropic import Anthropic

    client = Anthropic()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    for idx, frame in enumerate(batch, start=1):
        frame_path = str(frame.get("framePath") or "")
        transcript = str(frame.get("transcriptWindow") or "")
        content.append({
            "type": "text",
            "text": (
                f"Frame {idx}:\n"
                f"framePath: {frame_path}\n"
                f"videoId: {frame.get('videoId', '')}\n"
                f"timestamp: {frame.get('timestamp', '')}\n"
                f"sourceUrl: {frame.get('sourceUrl', '')}\n"
                f"transcriptWindow: {transcript}\n"
            ),
        })

        resolved = resolve_frame_path(frame_path)
        if resolved.exists():
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type(resolved),
                    "data": encode_image(resolved),
                },
            })

    content.append({"type": "text", "text": RESPONSE_INSTRUCTION})

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )

    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


# ─── OpenAI provider ──────────────────────────────────────────────────────

def call_openai(batch: list[dict[str, Any]], model: str, prompt: str) -> str:
    """Send a batch to OpenAI's vision API and return raw text response."""
    from openai import OpenAI

    client = OpenAI()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    for idx, frame in enumerate(batch, start=1):
        frame_path = str(frame.get("framePath") or "")
        transcript = str(frame.get("transcriptWindow") or "")
        content.append({
            "type": "text",
            "text": (
                f"Frame {idx}:\n"
                f"framePath: {frame_path}\n"
                f"videoId: {frame.get('videoId', '')}\n"
                f"timestamp: {frame.get('timestamp', '')}\n"
                f"sourceUrl: {frame.get('sourceUrl', '')}\n"
                f"transcriptWindow: {transcript}\n"
            ),
        })

        resolved = resolve_frame_path(frame_path)
        if resolved.exists():
            media = image_media_type(resolved)
            b64 = encode_image(resolved)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media};base64,{b64}",
                    "detail": "high",
                },
            })

    content.append({"type": "text", "text": RESPONSE_INSTRUCTION})

    # gpt-5+ requires max_completion_tokens instead of max_tokens
    token_kwarg = "max_completion_tokens" if "gpt-5" in model or "o3" in model or "o4" in model else "max_tokens"
    response = client.chat.completions.create(
        model=model,
        **{token_kwarg: 4096},
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )

    return (response.choices[0].message.content or "").strip()


# ─── Main ─────────────────────────────────────────────────────────────────

def find_payload_for_frame(frame: dict[str, Any], payloads: list[dict[str, Any]], index: int) -> dict[str, Any]:
    frame_path = str(frame.get("framePath") or "")

    for payload in payloads:
        if str(payload.get("framePath") or "") == frame_path:
            return payload

    for payload in payloads:
        payload_index = payload.get("frameIndex")
        try:
            if int(payload_index) == index:
                return payload
        except (TypeError, ValueError):
            continue

    if index - 1 < len(payloads):
        return payloads[index - 1]

    return {"category": "FILLER", "confidence": 0.0}


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    frames = load_json(input_path, default=[])

    if not isinstance(frames, list):
        raise ValueError(f"Expected list input: {input_path}")

    provider = detect_provider(args.provider)
    model = args.model or DEFAULT_MODELS[provider]
    LOGGER.info("Using provider=%s model=%s", provider, model)

    call_fn = call_anthropic if provider == "anthropic" else call_openai

    annotations: list[dict[str, Any]] = []
    kept = 0
    discarded = 0

    # Load existing checkpoint if present (resume-safe)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    processed_frame_paths: set[str] = set()
    if checkpoint_path.exists():
        try:
            existing = load_json(checkpoint_path, default=[])
            if isinstance(existing, list):
                annotations = existing
                for a in annotations:
                    fp = str(a.get("framePath") or "")
                    if fp:
                        processed_frame_paths.add(fp)
                kept = sum(1 for a in annotations if a.get("kept"))
                discarded = len(annotations) - kept
                LOGGER.info("Resuming from checkpoint: %d frames already processed", len(annotations))
        except Exception:
            pass

    # Filter out already-processed frames
    remaining_frames = [f for f in frames if str(f.get("framePath") or "") not in processed_frame_paths]

    for batch_idx, batch in enumerate(chunked(remaining_frames, max(1, args.batch_size))):
        title = str(batch[0].get("videoTitle") or "Unknown Video")
        channel = str(batch[0].get("channelName") or "Unknown Channel")
        prompt = PROMPT_TEMPLATE.format(title=title, channel=channel)

        LOGGER.info("Processing batch %d (%d frames) via %s/%s", batch_idx + 1, len(batch), provider, model)

        try:
            raw_text = call_fn(batch, model, prompt)
            parsed = parse_json_array(raw_text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Batch %d call failed: %s", batch_idx + 1, exc)
            parsed = []

        for index, frame in enumerate(batch, start=1):
            model_payload = find_payload_for_frame(frame, parsed, index)
            normalized = normalize_annotation(frame, model_payload, args.confidence_threshold)
            annotations.append(normalized)
            if normalized["kept"]:
                kept += 1
            else:
                discarded += 1

        # Checkpoint after every batch so progress is never lost
        save_json(checkpoint_path, annotations)

    save_json(output_path, annotations)
    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    LOGGER.info("Classification complete (total=%s kept=%s discarded=%s provider=%s)",
                len(annotations), kept, discarded, provider)
    print(f"Classified {len(annotations)} frames (kept {kept}, discarded {discarded}) via {provider}/{model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
