#!/usr/bin/env python3
"""Generate a daily markdown brief from frame annotations and watchlist context."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from _common import ensure_dir, load_json, resolve_path, setup_logging

LOGGER = logging.getLogger("content_scout.generate_brief")

# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-20250514",
    "openai": "gpt-5.2",
}

BRIEF_INSTRUCTIONS = """You are a YouTube content strategist writing a daily brief for Hans, an options trader who runs his own YouTube channel. Hans already has his own market intelligence — he does NOT need trading signals or price levels. He needs CONTENT IDEAS.

## Your job:
Analyze the frame annotations (with transcripts) from competitor YouTube videos. Tell Hans what content is being made, what's working, what's missing, and what HE should make next.

## Output format (markdown):

### 🎯 Video Ideas for Hans
The most important section. Based on everything analyzed today, give Hans 3-5 SPECIFIC video ideas he could film. For each:
- **Title** (written as an actual YouTube title — clickable, specific)
- **Hook** — the opening 15 seconds, how to grab attention
- **Angle** — what makes Hans's take different from what competitors already covered?
- **Why now** — why is this timely?
- **Inspired by** — [timestamp link] to the competitor moment that sparked this idea

### 📺 Competitor Breakdown
For each video analyzed today:
- **Video:** title + channel + [link]
- **Thesis:** What's the ONE argument this video is making? (1-2 sentences)
- **Format:** How is it structured? (prediction, recap, tutorial, reaction, etc.)
- **What works:** What's good about this video? What could Hans learn from it?
- **What's missing:** What did they leave out that Hans could cover?
- **Best moment:** [timestamp link] to the most interesting/useful part
- **Visual approach:** How did they use charts/graphics to support their points?

### 🔥 Trending Topics
What subjects are competitors all covering right now? Group by theme:
- Which topics are oversaturated (everyone's saying the same thing)?
- Which topics have room for a fresh take?
- Any emerging topics only 1 creator covered that could blow up?

### 💡 Content Techniques Worth Stealing
Specific presentation techniques, hooks, formats, or visual approaches that worked well:
- What it is, who did it, [timestamp link]
- How Hans could adapt it for his style

### 📊 Quick Stats
| Video | Channel | Topic | Format | Best Moment |
Summary table of everything analyzed.

## Rules:
- This is about CONTENT STRATEGY, not trading strategy.
- Every video idea must be specific enough that Hans could film it tomorrow.
- Quote what competitors said (verbal_context) to show what angles they took.
- Timestamp links for everything — Hans should be able to click and see the moment.
- If only one video was analyzed, be honest about limited data.
- Think like a YouTube strategist, not a trading analyst.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="tmp/annotations.json", help="Annotations JSON path")
    parser.add_argument("--transcripts-dir", default="tmp/transcripts",
                        help="Directory containing full transcript JSON files")
    parser.add_argument("--watchlist", default="config/watchlist.json",
                        help="Watchlist JSON path")
    parser.add_argument("--output", default=None,
                        help="Output markdown path (default: content-vault/daily/YYYY-MM-DD/_brief-YYYY-MM-DD.md)")
    parser.add_argument("--provider", choices=["anthropic", "openai", "auto"], default="auto",
                        help="LLM provider (default: auto-detect from available API keys)")
    parser.add_argument("--model", default=None, help="Model name (default: provider-specific)")
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


def load_transcripts(transcripts_dir: Path) -> dict[str, str]:
    """Load full transcripts from JSON files, keyed by video ID."""
    transcripts: dict[str, str] = {}
    if not transcripts_dir.exists():
        return transcripts
    for path in sorted(transcripts_dir.glob("*_transcript.json")):
        video_id = path.stem.replace("_transcript", "")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Whisper segments format
                full_text = " ".join(seg.get("text", "").strip() for seg in data if seg.get("text"))
            elif isinstance(data, dict):
                full_text = data.get("text", "")
            else:
                full_text = str(data)
            transcripts[video_id] = full_text
        except Exception:
            LOGGER.warning("Could not load transcript %s", path)
    return transcripts


def build_prompt(annotations: list[dict[str, Any]], watchlist: dict[str, Any],
                 transcripts: dict[str, str] | None = None) -> str:
    kept = [item for item in annotations if item.get("kept")]

    # Build compact annotation summary (skip redundant transcriptWindow to save tokens)
    compact_annotations = []
    for item in kept:
        compact = {k: v for k, v in item.items()
                   if k != "transcriptWindow" and k != "raw" and v is not None}
        compact_annotations.append(compact)

    parts = [BRIEF_INSTRUCTIONS, "\n"]

    if watchlist:
        parts.append("Watchlist JSON:\n")
        parts.append(json.dumps(watchlist, indent=2, ensure_ascii=False))
        parts.append("\n\n")

    # Include full transcripts for each video (much richer context than frame snippets)
    if transcripts:
        parts.append("FULL VIDEO TRANSCRIPTS (use these for understanding what was said):\n\n")
        for video_id, text in transcripts.items():
            # Find video title from annotations
            title = next((a.get("videoTitle", video_id) for a in kept if a.get("videoId") == video_id), video_id)
            channel = next((a.get("channelName", "?") for a in kept if a.get("videoId") == video_id), "?")
            parts.append(f"--- {channel}: {title} (ID: {video_id}) ---\n")
            # Truncate to ~4000 chars per transcript to manage token budget
            if len(text) > 4000:
                parts.append(text[:4000] + "\n[...transcript truncated...]\n\n")
            else:
                parts.append(text + "\n\n")

    parts.append("Frame Annotations (visual highlights only):\n")
    parts.append(json.dumps(compact_annotations, indent=2, ensure_ascii=False))
    parts.append("\n")

    return "".join(parts)


def call_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    is_reasoning = any(tag in model for tag in ("gpt-5", "o3", "o4"))
    # Reasoning models need much more headroom — internal reasoning tokens
    # count against max_completion_tokens. 16384 gives ~12K for reasoning
    # + ~4K for visible output.
    token_kwarg = "max_completion_tokens" if is_reasoning else "max_tokens"
    token_limit = 16384 if is_reasoning else 4096
    # Reasoning models ignore temperature (always 1), so only set for non-reasoning
    extra_kwargs = {} if is_reasoning else {"temperature": 0.2}
    response = client.chat.completions.create(
        model=model,
        **{token_kwarg: token_limit},
        **extra_kwargs,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").strip()
    # Log usage for debugging token budget issues
    usage = getattr(response, "usage", None)
    if usage:
        LOGGER.info("Token usage: prompt=%s completion=%s total=%s",
                     getattr(usage, "prompt_tokens", "?"),
                     getattr(usage, "completion_tokens", "?"),
                     getattr(usage, "total_tokens", "?"))
    return content


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    annotations_path = resolve_path(args.annotations)
    watchlist_path = resolve_path(args.watchlist)

    # Auto-generate dated output path if none specified
    if args.output is None:
        from datetime import date
        today = date.today().isoformat()
        args.output = f"content-vault/daily/{today}/_brief-{today}.md"

    output_path = resolve_path(args.output)

    annotations = load_json(annotations_path, default=[])
    watchlist = load_json(watchlist_path, default={})

    # Try primary transcripts dir (tmp/transcripts), fall back to archived copy
    transcripts_dir = resolve_path(args.transcripts_dir)
    transcripts = load_transcripts(transcripts_dir)
    if not transcripts:
        # Infer date from output path or annotations for fallback
        archive_date = None
        out_parts = str(args.output).split("/")
        for part in out_parts:
            if len(part) == 10 and part.count("-") == 2:
                archive_date = part
                break
        if archive_date:
            fallback_dir = resolve_path(f"content-vault/transcripts/{archive_date}")
            if fallback_dir.exists():
                transcripts = load_transcripts(fallback_dir)
                LOGGER.info("Using archived transcripts from %s (%d loaded)", fallback_dir, len(transcripts))
    LOGGER.info("Loaded %d full transcripts from %s", len(transcripts), transcripts_dir)

    if not isinstance(annotations, list):
        raise ValueError(f"Annotations file must contain a list: {annotations_path}")

    provider = detect_provider(args.provider)
    model = args.model or DEFAULT_MODELS[provider]
    LOGGER.info("Using provider=%s model=%s", provider, model)

    prompt = build_prompt(annotations, watchlist, transcripts=transcripts)

    call_fn = call_anthropic if provider == "anthropic" else call_openai

    try:
        markdown = call_fn(prompt, model)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Brief generation failed: %s", exc)
        markdown = ""

    if not markdown:
        markdown = "# Daily Brief\n\nNo content generated."

    ensure_dir(output_path.parent)
    output_path.write_text(markdown, encoding="utf-8")

    LOGGER.info("Daily brief written to %s via %s/%s", output_path, provider, model)
    print(f"Generated daily brief at {output_path} via {provider}/{model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
