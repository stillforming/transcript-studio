#!/usr/bin/env python3
"""Export transcript, visuals, and summary into a Transcript Studio Notion page."""

from __future__ import annotations

import argparse
import bisect
import logging
import os
import time
from datetime import date
from typing import Any
from urllib.parse import urljoin

from notion_client import Client

from _common import ContentScoutError, load_json, resolve_path, setup_logging

LOGGER = logging.getLogger("content_scout.export_notion")

MAX_RICH_TEXT_CHARS = 1900
MAX_BLOCKS_PER_APPEND = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to {video_id}_transcript.json produced by transcribe_local.py",
    )
    parser.add_argument(
        "--merged",
        required=True,
        help="Path to merged_transcript.json produced by merge_visuals.py",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to summary.json produced by summarize_video.py",
    )
    parser.add_argument(
        "--database-id",
        default=os.environ.get("NOTION_TRANSCRIPT_DB", ""),
        help="Notion Transcript Studio database ID",
    )
    parser.add_argument("--token", default=os.environ.get("NOTION_TOKEN", ""), help="Notion API token")
    parser.add_argument(
        "--image-base-url",
        default="",
        help="Optional URL prefix for visual image_path values (external Notion image embeds)",
    )
    parser.add_argument("--delay", type=float, default=0.35, help="Rate limit delay between API calls")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def normalize_text(value: Any, fallback: str = "") -> str:
    return str(value or fallback).strip()


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_timestamp(seconds: int) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def split_text_chunks(content: str, max_chars: int = MAX_RICH_TEXT_CHARS) -> list[str]:
    text = str(content or "")
    if not text:
        return []

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        split_at = max(
            remaining.rfind("\n", 0, max_chars + 1),
            remaining.rfind(" ", 0, max_chars + 1),
        )
        if split_at <= 0:
            split_at = max_chars

        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:max_chars]
            split_at = max_chars

        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    return chunks


def to_plain_text_objects(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": chunk}} for chunk in split_text_chunks(content)]


def to_rich_text_objects(content: str, *, bold: bool = False) -> list[dict[str, Any]]:
    rich: list[dict[str, Any]] = []
    for chunk in split_text_chunks(content):
        text_obj: dict[str, Any] = {"type": "text", "text": {"content": chunk}}
        if bold:
            text_obj["annotations"] = {"bold": True}
        rich.append(text_obj)
    return rich


def to_title(content: str) -> list[dict[str, Any]]:
    title = normalize_text(content, "Untitled")
    return to_plain_text_objects(title)[:1]


def to_property_rich_text(content: str) -> list[dict[str, Any]]:
    value = normalize_text(content)
    return to_plain_text_objects(value) if value else []


def paragraph_block(content: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": to_rich_text_objects(content)},
    }


def paragraph_rich_text_block(rich_text: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": rich_text},
    }


def speaker_paragraph_block(speaker: str, content: str) -> dict[str, Any]:
    rich_text = to_rich_text_objects(f"{speaker}: ", bold=True) + to_rich_text_objects(content)
    return paragraph_rich_text_block(rich_text)


def heading_block(
    level: int,
    content: str,
    *,
    is_toggleable: bool = False,
    children: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    heading_type = f"heading_{level}"
    payload: dict[str, Any] = {
        "object": "block",
        "type": heading_type,
        heading_type: {
            "rich_text": to_rich_text_objects(content),
            "is_toggleable": is_toggleable,
        },
    }
    if children:
        payload[heading_type]["children"] = children
    return payload


def heading_rich_text_block(
    level: int,
    rich_text: list[dict[str, Any]],
    *,
    is_toggleable: bool = False,
    children: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    heading_type = f"heading_{level}"
    payload: dict[str, Any] = {
        "object": "block",
        "type": heading_type,
        heading_type: {
            "rich_text": rich_text if rich_text else to_rich_text_objects(""),
            "is_toggleable": is_toggleable,
        },
    }
    if children:
        payload[heading_type]["children"] = children
    return payload


def callout_block(content: str, *, emoji: str = "ℹ️") -> dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": to_rich_text_objects(content),
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }


def callout_rich_text_block(rich_text: list[dict[str, Any]], *, emoji: str = "ℹ️") -> dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": rich_text if rich_text else to_rich_text_objects(""),
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }


def numbered_list_item_block(content: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": to_rich_text_objects(content)},
    }


def bulleted_list_item_block(content: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": to_rich_text_objects(content)},
    }


def bulleted_list_item_rich_text_block(rich_text: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": rich_text if rich_text else to_rich_text_objects("")},
    }


def image_block(url: str, *, caption: str = "") -> dict[str, Any]:
    image_payload: dict[str, Any] = {"type": "external", "external": {"url": url}}
    if caption:
        image_payload["caption"] = to_rich_text_objects(caption)
    return {
        "object": "block",
        "type": "image",
        "image": image_payload,
    }


def divider_block() -> dict[str, Any]:
    return {"object": "block", "type": "divider", "divider": {}}


def table_row_block(cells: list[str]) -> dict[str, Any]:
    normalized_cells: list[list[dict[str, Any]]] = []
    for cell in cells:
        rich = to_rich_text_objects(cell)
        if not rich:
            rich = [{"type": "text", "text": {"content": ""}}]
        normalized_cells.append(rich)
    return {
        "object": "block",
        "type": "table_row",
        "table_row": {"cells": normalized_cells},
    }


def table_block(rows: list[list[str]]) -> dict[str, Any]:
    table_rows = [table_row_block(["Timestamp", "Type", "Description"])]
    table_rows.extend(table_row_block(row) for row in rows)
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": 3,
            "has_column_header": True,
            "children": table_rows,
        },
    }


def safe_call(action: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Notion %s failed: %s", action, exc)
        raise


def normalize_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        return []

    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = normalize_text(raw.get("text"))
        if not text:
            continue
        start = max(0.0, coerce_float(raw.get("start"), 0.0))
        end = max(start, coerce_float(raw.get("end"), start))
        speaker = normalize_text(raw.get("speaker"), "Speaker 1")
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
        )
    segments.sort(key=lambda item: item["start"])
    return segments


def duration_minutes(segments: list[dict[str, Any]]) -> float:
    max_end = max((coerce_float(segment.get("end"), 0.0) for segment in segments), default=0.0)
    return round(max_end / 60.0, 2)


def speaker_count(payload: dict[str, Any], segments: list[dict[str, Any]]) -> int:
    from_payload = payload.get("speakers", [])
    if isinstance(from_payload, list):
        seen = {normalize_text(speaker) for speaker in from_payload if normalize_text(speaker)}
        if seen:
            return len(seen)

    seen_from_segments = {
        normalize_text(segment.get("speaker"), "Speaker 1")
        for segment in segments
        if normalize_text(segment.get("speaker"), "Speaker 1")
    }
    return len(seen_from_segments)


def build_page_metadata(payload: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "video_title": normalize_text(payload.get("videoTitle"), "Untitled Video"),
        "channel_name": normalize_text(payload.get("channelName"), "Unknown"),
        "source_url": normalize_text(payload.get("url")),
        "video_id": normalize_text(payload.get("videoId")),
        "duration_minutes": duration_minutes(segments),
        "speakers": speaker_count(payload, segments),
        "today": date.today().isoformat(),
        "preset": normalize_text(payload.get("preset")),
        "playlist": normalize_text(payload.get("playlist")),
        "tags": payload.get("tags", []),
    }


def build_page_properties(metadata: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "Name": {"title": to_title(f"{metadata['video_title']} — {metadata['channel_name']}")},
        "Date": {"date": {"start": metadata["today"]}},
        "Channel": {"select": {"name": metadata["channel_name"][:100]}},
        "Status": {"select": {"name": "Ready"}},
        "Source URL": {"url": metadata["source_url"] or None},
        "Video ID": {"rich_text": to_property_rich_text(metadata["video_id"])},
        "Duration": {"number": metadata["duration_minutes"]},
        "Speakers": {"number": metadata["speakers"]},
    }

    preset = normalize_text(metadata.get("preset"))
    if preset:
        properties["Preset"] = {"select": {"name": preset[:100]}}

    playlist = normalize_text(metadata.get("playlist"))
    if playlist:
        properties["Playlist"] = {"select": {"name": playlist[:100]}}

    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, list):
        tags = [normalize_text(tag)[:100] for tag in raw_tags if normalize_text(tag)]
        if tags:
            properties["Tags"] = {"multi_select": [{"name": tag} for tag in tags]}

    return properties


def build_metadata_callout(metadata: dict[str, Any]) -> dict[str, Any]:
    source_line = f"Source: {metadata['source_url']}" if metadata["source_url"] else "Source: n/a"
    info_line = (
        f"Speakers: {metadata['speakers']} | Duration: {metadata['duration_minutes']} minutes"
    )
    date_line = f"Date: {metadata['today']}"
    preset = normalize_text(metadata.get("preset"))
    preset_line = f"Preset: {preset}" if preset else "Preset: n/a"
    content = "\n".join([source_line, info_line, date_line, preset_line])
    return callout_block(content, emoji="📺")


def timestamp_from_summary_item(item: dict[str, Any], label_key: str, seconds_key: str) -> str:
    label = normalize_text(item.get(label_key))
    if label:
        return label
    seconds = int(max(0.0, coerce_float(item.get(seconds_key), 0.0)))
    return format_timestamp(seconds)


def parse_timestamp_seconds(value: Any) -> int:
    as_text = normalize_text(value)
    if not as_text:
        return 0
    if as_text.isdigit():
        return int(as_text)

    parts = as_text.split(":")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        minutes, seconds = int(parts[0]), int(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    return 0


def chapter_rich_text(timestamp: str, title: str) -> list[dict[str, Any]]:
    return to_rich_text_objects(timestamp, bold=True) + to_rich_text_objects(f" — {title}")


def normalize_sentence(text: str) -> str:
    clean = normalize_text(text)
    if not clean:
        return ""
    if clean.endswith((".", "!", "?")):
        return clean
    return f"{clean}."


def build_tldr(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    sentences: list[str] = []
    raw_takeaways = summary_payload.get("takeaways", [])
    if isinstance(raw_takeaways, list):
        for item in raw_takeaways[:3]:
            if not isinstance(item, dict):
                continue
            sentence = normalize_sentence(normalize_text(item.get("text")))
            if sentence:
                sentences.append(sentence)

    if not sentences:
        fallback = "No summary takeaways were generated for this video."
    else:
        fallback = " ".join(sentences)

    return [
        heading_block(2, "TL;DR"),
        paragraph_block(fallback),
    ]


def score_bar(value: int, max_val: int = 5) -> str:
    """Render a score as ●○○○○ (filled/empty circles)."""
    clamped = max(0, min(max_val, int(value)))
    return "●" * clamped + "○" * (max_val - clamped)


def build_signal_extraction_blocks(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the 🔥 Signal Extraction Layer that sits above TL;DR."""
    signal = summary_payload.get("signal_extraction")
    if not isinstance(signal, dict):
        return []

    blocks: list[dict[str, Any]] = []

    # ── Section heading ──
    blocks.append(heading_block(1, "🔥 Signal Extraction Layer"))

    # ── Quick Verdict ──
    verdict = normalize_text(signal.get("quick_verdict"), "No signal analysis available.")
    blocks.append(callout_block(verdict, emoji="🎯"))

    # ── Speaker Assessment ──
    speaker = normalize_text(signal.get("speaker_assessment"))
    if speaker and speaker != "No speaker assessment available.":
        blocks.append(callout_block(speaker, emoji="🔍"))

    # ── Scores Dashboard ──
    scores = signal.get("scores") or {}
    score_parts = [
        f"Macro: {score_bar(scores.get('macro_impact', 0))} {scores.get('macro_impact', 0)}",
        f"Ideas: {score_bar(scores.get('stock_idea_density', 0))} {scores.get('stock_idea_density', 0)}",
        f"Contrarian: {score_bar(scores.get('contrarian_value', 0))} {scores.get('contrarian_value', 0)}",
        f"AI/Infra: {score_bar(scores.get('ai_infrastructure_relevance', 0))} {scores.get('ai_infrastructure_relevance', 0)}",
        f"Show: {score_bar(scores.get('show_utility', 0))} {scores.get('show_utility', 0)}",
    ]
    blocks.append(callout_block(" · ".join(score_parts), emoji="📊"))

    # ── Market Bias ──
    bias = signal.get("market_bias") or {}
    blocks.append(heading_block(3, "Market Bias"))
    blocks.append(
        bulleted_list_item_block(f"Tone: {normalize_text(bias.get('tone'), '—')}")
    )
    blocks.append(
        bulleted_list_item_block(f"Regime: {normalize_text(bias.get('regime'), '—')}")
    )
    blocks.append(
        bulleted_list_item_block(f"Risk Tilt: {normalize_text(bias.get('risk_tilt'), '—')}")
    )
    blocks.append(
        bulleted_list_item_block(
            f"Cycle Position: {normalize_text(bias.get('cycle_position'), '—')}"
        )
    )

    # ── Macro Signals ──
    macro = signal.get("macro_signals") or {}
    blocks.append(heading_block(3, "Macro Signals"))
    has_signal = macro.get("has_signal", False)
    if not has_signal:
        blocks.append(
            paragraph_block("No meaningful macro signal in this video.")
        )
    else:
        for key in ("rates", "credit", "liquidity", "volatility", "structural_shifts"):
            value = normalize_text(macro.get(key))
            if value:
                label = key.replace("_", " ").title()
                blocks.append(bulleted_list_item_block(f"{label}: {value}"))

    # ── Ticker Mentions ──
    tickers = signal.get("tickers") or []
    blocks.append(heading_block(3, "Ticker Mentions"))
    if not tickers:
        blocks.append(paragraph_block("No tickers mentioned in this video."))
    else:
        ticker_rows: list[list[str]] = []
        for t in tickers:
            if not isinstance(t, dict):
                continue
            strength = int(t.get("signal_strength", 0))
            ticker_rows.append(
                [
                    normalize_text(t.get("ticker"), "—"),
                    normalize_text(t.get("direction"), "—"),
                    normalize_text(t.get("context"), "—"),
                    normalize_text(t.get("time_horizon"), "—"),
                    normalize_text(t.get("timestamp"), "—"),
                    score_bar(strength, 5),
                ]
            )
        if ticker_rows:
            # 6-column table: Ticker | Direction | Context | Horizon | Timestamp | Strength
            header = table_row_block(
                ["Ticker", "Direction", "Context", "Horizon", "Timestamp", "Strength"]
            )
            data_rows = [table_row_block(row) for row in ticker_rows]
            blocks.append(
                {
                    "object": "block",
                    "type": "table",
                    "table": {
                        "table_width": 6,
                        "has_column_header": True,
                        "children": [header] + data_rows,
                    },
                }
            )
        else:
            blocks.append(paragraph_block("No tickers mentioned in this video."))

    # ── Catalysts ──
    catalysts = signal.get("catalysts") or []
    blocks.append(heading_block(3, "Catalysts"))
    if not catalysts:
        blocks.append(paragraph_block("No specific market catalysts identified."))
    else:
        for cat in catalysts:
            text = normalize_text(cat)
            if text:
                blocks.append(bulleted_list_item_block(text))

    # ── Show Relevance ──
    relevance = signal.get("show_relevance") or {}
    blocks.append(heading_block(3, "Show Relevance"))
    for key, label in (
        ("ai_thesis", "AI 2.0 Thesis"),
        ("software_repricing", "Software Repricing"),
        ("infrastructure", "Infrastructure"),
        ("rate_regime", "Rate Regime"),
        ("liquidity_cycle", "Liquidity Cycle"),
    ):
        blocks.append(
            bulleted_list_item_block(
                f"{label}: {normalize_text(relevance.get(key), '—')}"
            )
        )
    contradicts = normalize_text(relevance.get("contradicts_narrative"))
    blocks.append(
        bulleted_list_item_block(
            f"Contradicts Narrative: {contradicts if contradicts else '—'}"
        )
    )
    segment = normalize_text(relevance.get("segment_potential"))
    if segment:
        blocks.append(
            bulleted_list_item_block(f"Segment Potential: {segment}")
        )

    # ── Divider to separate signal from content ──
    blocks.append(divider_block())

    return blocks


def summary_takeaway_blocks(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    raw_takeaways = summary_payload.get("takeaways", [])
    if isinstance(raw_takeaways, list):
        for item in raw_takeaways:
            if not isinstance(item, dict):
                continue
            text = normalize_text(item.get("text"))
            if not text:
                continue
            timestamp = timestamp_from_summary_item(item, "timestamp", "timestamp_seconds")
            blocks.append(numbered_list_item_block(f"{text} ({timestamp})"))
    if not blocks:
        blocks.append(numbered_list_item_block("No key takeaways generated."))
    return blocks


def summary_chapter_blocks(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    raw_chapters = summary_payload.get("chapters", [])
    if isinstance(raw_chapters, list):
        for item in raw_chapters:
            if not isinstance(item, dict):
                continue
            title = normalize_text(item.get("title"), normalize_text(item.get("text")))
            if not title:
                continue
            timestamp = timestamp_from_summary_item(item, "timestamp", "timestamp_seconds")
            blocks.append(bulleted_list_item_rich_text_block(chapter_rich_text(timestamp, title)))
    if not blocks:
        blocks.append(bulleted_list_item_block("No chapters generated."))
    return blocks


def short_callout(item: dict[str, Any]) -> dict[str, Any] | None:
    start = timestamp_from_summary_item(item, "start", "start_seconds")
    end = timestamp_from_summary_item(item, "end", "end_seconds")
    hook = normalize_text(item.get("hook"), "n/a")
    payoff = normalize_text(item.get("payoff"), "n/a")

    if hook == "n/a" and payoff == "n/a":
        return None

    rich_text = (
        to_rich_text_objects("Hook: ", bold=True)
        + to_rich_text_objects(hook)
        + to_rich_text_objects("\n")
        + to_rich_text_objects("Payoff: ", bold=True)
        + to_rich_text_objects(payoff)
        + to_rich_text_objects(f"\n⏱️ {start} → {end}")
    )
    return callout_rich_text_block(rich_text, emoji="🎬")


def summary_shorts_children(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    raw_shorts = summary_payload.get("shorts", [])
    if isinstance(raw_shorts, list):
        for item in raw_shorts:
            if not isinstance(item, dict):
                continue
            short_block = short_callout(item)
            if short_block:
                children.append(short_block)

    if not children:
        children.append(callout_block("No shorts/clip candidates generated.", emoji="🎬"))
    return children


def summary_slide_children(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    raw_slides = summary_payload.get("slide_suggestions", summary_payload.get("slides", []))
    if isinstance(raw_slides, list):
        for item in raw_slides:
            if not isinstance(item, dict):
                continue
            text = normalize_text(item.get("text"), normalize_text(item.get("suggestion")))
            if not text:
                continue
            timestamp = normalize_text(item.get("timestamp"))
            if timestamp:
                children.append(bulleted_list_item_block(f"{text} ({timestamp})"))
            else:
                children.append(bulleted_list_item_block(text))
    if not children:
        children.append(bulleted_list_item_block("No slide/graphic notes generated."))
    return children


def build_summary_blocks(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [heading_block(2, "🎯 Key Takeaways")]
    blocks.extend(summary_takeaway_blocks(summary_payload))
    blocks.append(heading_block(2, "📑 Chapters"))
    blocks.extend(summary_chapter_blocks(summary_payload))
    blocks.append(
        heading_block(
            2,
            "🎬 Shorts & Clip Ideas",
            is_toggleable=True,
            children=summary_shorts_children(summary_payload),
        )
    )
    blocks.append(
        heading_block(
            2,
            "📊 Slide & Graphic Notes",
            is_toggleable=True,
            children=summary_slide_children(summary_payload),
        )
    )
    return blocks


def merged_value_dict(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("value")
    return value if isinstance(value, dict) else {}


def merged_attr(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)

    nested = merged_value_dict(item)
    for key in keys:
        if nested.get(key) is not None:
            return nested.get(key)
    return None


def seconds_from_value(value: Any) -> int:
    if isinstance(value, str) and ":" in value:
        return parse_timestamp_seconds(value)
    return int(max(0.0, coerce_float(value, 0.0)))


def visual_timestamp(item: dict[str, Any]) -> str:
    label = normalize_text(merged_attr(item, "timestamp_label", "label"))
    if label:
        return label

    timestamp = merged_attr(item, "timestamp")
    as_text = normalize_text(timestamp)
    if as_text and ":" in as_text:
        return as_text
    if as_text.isdigit():
        return format_timestamp(int(as_text))

    seconds = seconds_from_value(merged_attr(item, "timestamp_seconds", "seconds", "start"))
    return format_timestamp(seconds)


def visual_type(item: dict[str, Any]) -> str:
    for key in ("visual_type", "visualType", "label", "category"):
        value = normalize_text(merged_attr(item, key))
        if value:
            return value
    return "Visual"


def resolve_image_url(base_url: str, image_path: str) -> str:
    clean_base = normalize_text(base_url)
    clean_path = normalize_text(image_path)
    if not clean_base:
        return ""
    if not clean_path:
        return ""
    if clean_path.startswith(("http://", "https://")):
        return clean_path
    if clean_path.startswith("./"):
        clean_path = clean_path[2:]
    # Strip common local prefixes (tmp/frames/, frames/, etc.)
    for prefix in ("tmp/frames/", "frames/", "tmp/"):
        if clean_path.startswith(prefix):
            clean_path = clean_path[len(prefix):]
            break
    return urljoin(clean_base.rstrip("/") + "/", clean_path.lstrip("/"))


def normalize_chapters(chapters: Any) -> list[dict[str, Any]]:
    raw_chapters = chapters if isinstance(chapters, list) else []
    normalized: list[tuple[int, int, str, str]] = []

    for index, item in enumerate(raw_chapters):
        if not isinstance(item, dict):
            continue

        title = normalize_text(item.get("title"), normalize_text(item.get("text")))
        if not title:
            title = f"Chapter {len(normalized) + 1}"

        raw_seconds = merged_attr(item, "timestamp_seconds", "seconds")
        seconds = seconds_from_value(raw_seconds)
        if seconds == 0:
            seconds = parse_timestamp_seconds(item.get("timestamp"))

        timestamp = normalize_text(item.get("timestamp"))
        if not timestamp:
            timestamp = format_timestamp(seconds)

        normalized.append((seconds, index, timestamp, title))

    normalized.sort(key=lambda row: (row[0], row[1]))

    deduped: list[dict[str, Any]] = []
    seen_seconds: set[int] = set()
    for seconds, _, timestamp, title in normalized:
        if seconds in seen_seconds:
            continue
        seen_seconds.add(seconds)
        deduped.append(
            {
                "seconds": seconds,
                "timestamp": timestamp,
                "title": title,
            }
        )

    if not deduped:
        return [{"seconds": 0, "timestamp": "00:00", "title": "Opening"}]

    if deduped[0]["seconds"] > 0:
        deduped.insert(0, {"seconds": 0, "timestamp": "00:00", "title": "Opening"})
    return deduped


def merge_text_fragments(fragments: list[str]) -> str:
    merged = ""
    no_space_before = (",", ".", "!", "?", ";", ":", ")", "]")
    no_space_after = ("-", "—", "–", "/")

    for fragment in fragments:
        piece = normalize_text(fragment)
        if not piece:
            continue
        if not merged:
            merged = piece
            continue
        if merged.endswith(no_space_after) or piece.startswith(no_space_before):
            merged += piece
        else:
            merged += f" {piece}"
    return merged


def visual_to_block(item: dict[str, Any], image_base_url: str) -> dict[str, Any]:
    timestamp = visual_timestamp(item)
    v_type = visual_type(item)
    description = normalize_text(merged_attr(item, "description"), "No description")
    caption = f"{timestamp} — {v_type}: {description}"

    image_path = normalize_text(
        merged_attr(item, "image_path", "imagePath", "framePath", "frame")
    )
    image_url = resolve_image_url(image_base_url, image_path)
    if image_url:
        return image_block(image_url, caption=caption)
    return callout_block(caption, emoji="🖼️")


def collect_visual_rows(merged_payload: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in merged_payload:
        if not isinstance(item, dict):
            continue
        if normalize_text(merged_attr(item, "type")).lower() != "visual":
            continue
        timestamp = visual_timestamp(item)
        v_type = visual_type(item)
        description = normalize_text(merged_attr(item, "description"), "No description")
        rows.append([timestamp, v_type, description])
    return rows


def merge_transcript_paragraphs(
    merged_blocks: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    speaker_total: int,
    image_base_url: str = "",
) -> list[dict[str, Any]]:
    chapter_defs = normalize_chapters(chapters)
    chapter_starts = [chapter["seconds"] for chapter in chapter_defs]
    events_by_chapter: list[list[dict[str, Any]]] = [[] for _ in chapter_defs]

    for item in merged_blocks:
        if not isinstance(item, dict):
            continue
        block_type = normalize_text(merged_attr(item, "type")).lower()
        if block_type == "timestamp":
            continue

        if block_type == "text":
            text = normalize_text(merged_attr(item, "text", "content"))
            if not text:
                continue
            speaker = normalize_text(merged_attr(item, "speaker"), "Speaker 1")
            start_seconds = seconds_from_value(merged_attr(item, "start", "timestamp", "seconds"))
            chapter_index = max(0, bisect.bisect_right(chapter_starts, start_seconds) - 1)
            chapter_index = min(chapter_index, len(events_by_chapter) - 1)
            events_by_chapter[chapter_index].append(
                {
                    "type": "text",
                    "speaker": speaker,
                    "text": text,
                }
            )
            continue

        if block_type != "visual":
            continue

        visual_seconds = seconds_from_value(merged_attr(item, "timestamp_seconds", "timestamp", "seconds"))
        chapter_index = max(0, bisect.bisect_right(chapter_starts, visual_seconds) - 1)
        chapter_index = min(chapter_index, len(events_by_chapter) - 1)
        events_by_chapter[chapter_index].append(item)

    blocks: list[dict[str, Any]] = []
    multi_speaker = speaker_total > 1

    for chapter, chapter_events in zip(chapter_defs, events_by_chapter):
        blocks.append(
            heading_rich_text_block(
                3,
                chapter_rich_text(chapter["timestamp"], chapter["title"]),
            )
        )

        turn_speaker = ""
        turn_fragments: list[str] = []

        def flush_turn() -> None:
            nonlocal turn_speaker, turn_fragments
            if not turn_fragments:
                return
            text = merge_text_fragments(turn_fragments)
            if not text:
                turn_fragments = []
                turn_speaker = ""
                return
            if multi_speaker:
                blocks.append(speaker_paragraph_block(turn_speaker, text))
            else:
                blocks.append(paragraph_block(text))
            turn_fragments = []
            turn_speaker = ""

        for event in chapter_events:
            event_type = normalize_text(merged_attr(event, "type")).lower()
            if event_type == "text":
                speaker = normalize_text(merged_attr(event, "speaker"), "Speaker 1")
                text = normalize_text(merged_attr(event, "text"))
                if not text:
                    continue
                if not turn_fragments:
                    turn_speaker = speaker
                    turn_fragments = [text]
                    continue
                if speaker == turn_speaker:
                    turn_fragments.append(text)
                    continue

                flush_turn()
                turn_speaker = speaker
                turn_fragments = [text]
                continue

            if event_type == "visual":
                flush_turn()
                blocks.append(visual_to_block(event, image_base_url))

        flush_turn()

    return blocks


def build_transcript_blocks(
    merged_payload: list[dict[str, Any]],
    image_base_url: str,
    chapters: list[dict[str, Any]] | None = None,
    speaker_total: int = 1,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    visual_rows = collect_visual_rows(merged_payload)
    chapter_source = chapters if isinstance(chapters, list) else []
    transcript_children = merge_transcript_paragraphs(
        merged_payload,
        chapter_source,
        speaker_total,
        image_base_url,
    )

    blocks: list[dict[str, Any]] = [heading_block(1, "Full Transcript")]
    if transcript_children:
        blocks.extend(transcript_children)
    else:
        blocks.append(paragraph_block("No merged transcript blocks were found."))
    return blocks, visual_rows


def build_visual_index_toggle(visual_rows: list[list[str]]) -> dict[str, Any]:
    rows = visual_rows if visual_rows else [["--", "--", "No visuals found."]]
    # Notion tables are limited to 100 children (header + 99 data rows).
    # Truncate if needed to stay within the limit.
    max_data_rows = 98  # 1 header row + 98 data rows = 99 < 100
    if len(rows) > max_data_rows:
        rows = rows[:max_data_rows]
    return heading_block(
        1,
        "Visual Index",
        is_toggleable=True,
        children=[table_block(rows)],
    )


def split_raw_transcript(raw_text: str) -> list[str]:
    clean = raw_text.replace("\r\n", "\n").strip()
    if not clean:
        return []

    chunks: list[str] = []
    for paragraph in clean.split("\n\n"):
        para = paragraph.strip()
        if not para:
            continue
        chunks.extend(split_text_chunks(para))
    if chunks:
        return chunks
    return split_text_chunks(clean)


def merge_raw_segments(
    segments: list[dict[str, Any]],
    *,
    multi_speaker: bool,
    max_window_seconds: int = 60,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    children: list[dict[str, Any]] = []
    turn_speaker = ""
    turn_start = 0
    turn_fragments: list[str] = []

    def flush_turn() -> None:
        nonlocal turn_speaker, turn_start, turn_fragments
        if not turn_fragments:
            return
        merged_text = merge_text_fragments(turn_fragments)
        if merged_text:
            if multi_speaker:
                children.append(speaker_paragraph_block(turn_speaker, merged_text))
            else:
                children.append(paragraph_block(merged_text))
        turn_speaker = ""
        turn_start = 0
        turn_fragments = []

    for segment in segments:
        text = normalize_text(segment.get("text"))
        if not text:
            continue

        speaker = normalize_text(segment.get("speaker"), "Speaker 1")
        start_seconds = int(max(0.0, coerce_float(segment.get("start"), 0.0)))

        if not turn_fragments:
            turn_speaker = speaker
            turn_start = start_seconds
            turn_fragments = [text]
            continue

        speaker_changed = speaker != turn_speaker
        window_exceeded = (start_seconds - turn_start) >= max_window_seconds
        if speaker_changed or window_exceeded:
            flush_turn()
            turn_speaker = speaker
            turn_start = start_seconds
            turn_fragments = [text]
            continue

        turn_fragments.append(text)

    flush_turn()
    return children


def build_raw_transcript_toggle(payload: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    children = merge_raw_segments(
        segments,
        multi_speaker=speaker_count(payload, segments) > 1,
        max_window_seconds=60,
    )

    if not children:
        raw_text = normalize_text(payload.get("raw_text"), normalize_text(payload.get("text")))
        if not raw_text:
            raw_text = " ".join(
                normalize_text(segment.get("text"))
                for segment in segments
                if normalize_text(segment.get("text"))
            )
        chunks = split_raw_transcript(raw_text)
        children = [paragraph_block(chunk) for chunk in chunks]
        if not children:
            children = [paragraph_block("No raw transcript text available.")]

    return heading_block(1, "Raw Transcript", is_toggleable=True, children=children)


def build_page_blocks(
    transcript_payload: dict[str, Any],
    merged_payload: list[dict[str, Any]],
    summary_payload: dict[str, Any],
    image_base_url: str,
) -> list[dict[str, Any]]:
    segments = normalize_segments(transcript_payload)
    metadata = build_page_metadata(transcript_payload, segments)

    blocks: list[dict[str, Any]] = [build_metadata_callout(metadata)]
    blocks.extend(build_signal_extraction_blocks(summary_payload))
    blocks.extend(build_tldr(summary_payload))
    blocks.extend(build_summary_blocks(summary_payload))

    transcript_blocks, visual_rows = build_transcript_blocks(
        merged_payload,
        image_base_url,
        chapters=summary_payload.get("chapters", []),
        speaker_total=metadata["speakers"],
    )
    blocks.extend(transcript_blocks)
    blocks.append(build_visual_index_toggle(visual_rows))
    blocks.append(build_raw_transcript_toggle(transcript_payload, segments))
    return blocks


def append_blocks(notion: Client, page_id: str, blocks: list[dict[str, Any]], delay: float) -> int:
    failed_calls = 0
    for start in range(0, len(blocks), MAX_BLOCKS_PER_APPEND):
        chunk = blocks[start : start + MAX_BLOCKS_PER_APPEND]
        try:
            safe_call(
                "children append",
                notion.blocks.children.append,
                block_id=page_id,
                children=chunk,
            )
        except Exception:
            failed_calls += 1
        time.sleep(delay)
    return failed_calls


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        if not args.database_id:
            raise RuntimeError("--database-id or NOTION_TRANSCRIPT_DB is required")
        if not args.token:
            raise RuntimeError("--token or NOTION_TOKEN is required")

        transcript_path = resolve_path(args.transcript)
        merged_path = resolve_path(args.merged)
        summary_path = resolve_path(args.summary)

        transcript_payload = load_json(transcript_path, default={})
        merged_payload = load_json(merged_path, default=[])
        summary_payload = load_json(summary_path, default={})

        if not isinstance(transcript_payload, dict):
            raise ContentScoutError(f"Transcript payload must be an object: {transcript_path}")
        if not isinstance(merged_payload, list):
            raise ContentScoutError(f"Merged payload must be an array: {merged_path}")
        if not isinstance(summary_payload, dict):
            raise ContentScoutError(f"Summary payload must be an object: {summary_path}")

        segments = normalize_segments(transcript_payload)
        metadata = build_page_metadata(transcript_payload, segments)
        properties = build_page_properties(metadata)

        notion = Client(auth=args.token)
        page = safe_call(
            "page create",
            notion.pages.create,
            parent={"database_id": args.database_id},
            properties=properties,
        )
        page_id = normalize_text(page.get("id"))
        if not page_id:
            raise RuntimeError("Notion page create returned no page id")
        time.sleep(args.delay)
        LOGGER.info("Created Notion page %s", page_id)

        try:
            blocks = build_page_blocks(
                transcript_payload=transcript_payload,
                merged_payload=merged_payload,
                summary_payload=summary_payload,
                image_base_url=args.image_base_url,
            )
            append_failures = append_blocks(notion, page_id, blocks, args.delay)
            if append_failures:
                LOGGER.error(
                    "Page %s created, but %s children.append call(s) failed.",
                    page_id,
                    append_failures,
                )
                print(
                    f"Created Notion page {page_id}; block append had "
                    f"{append_failures} failed call(s)."
                )
                return 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed appending content to page %s: %s", page_id, exc)
            print(f"Created Notion page {page_id}; failed to append full content.")
            return 1

        print(f"Created Notion page {page_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("export_notion failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
