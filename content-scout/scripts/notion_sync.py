#!/usr/bin/env python3
"""Sync kept annotations into a Notion Content Vault database."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import date
from typing import Any

from notion_client import Client

from _common import load_json, resolve_path, setup_logging

LOGGER = logging.getLogger("content_scout.notion_sync")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="tmp/annotations.json", help="Annotations JSON path")
    parser.add_argument("--database-id", default=os.environ.get("NOTION_CONTENT_VAULT_DB", ""), help="Notion database ID")
    parser.add_argument("--token", default=os.environ.get("NOTION_TOKEN", ""), help="Notion API token")
    parser.add_argument("--delay", type=float, default=0.35, help="Rate limit delay between API calls")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def to_title(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": content[:2000]}}]


def to_rich_text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": content[:2000]}}] if content else []


def mmss(seconds: int) -> str:
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def normalize_text(value: Any, fallback: str = "") -> str:
    text = str(value or fallback).strip()
    return text


def normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out = []
    for item in values:
        text = normalize_text(item)
        if text:
            out.append(text)
    return out


def group_kept_annotations(annotations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        if not isinstance(item, dict) or not item.get("kept"):
            continue
        video_id = normalize_text(item.get("videoId"))
        if not video_id:
            continue
        grouped[video_id].append(item)
    return grouped


def build_page_properties(video_annotations: list[dict[str, Any]]) -> dict[str, Any]:
    first = video_annotations[0]
    title = normalize_text(first.get("videoTitle"), "Untitled Video")
    channel = normalize_text(first.get("channelName"), "Unknown")
    video_id = normalize_text(first.get("videoId"))
    source_url = normalize_text(first.get("sourceUrl"), "")

    tags = sorted(
        {
            tag
            for item in video_annotations
            for tag in normalize_list(item.get("tags", []))
            if tag
        }
    )
    tickers = sorted(
        {
            normalize_text(item.get("ticker"))
            for item in video_annotations
            if normalize_text(item.get("ticker"))
        }
    )

    relevance_values = [
        int(item.get("relevance"))
        for item in video_annotations
        if isinstance(item.get("relevance"), int)
    ]
    avg_relevance = round(sum(relevance_values) / len(relevance_values), 2) if relevance_values else 0

    return {
        "Name": {"title": to_title(f"{title} - {channel}")},
        "Date": {"date": {"start": date.today().isoformat()}},
        "Channel": {"select": {"name": channel[:100]}},
        "Tickers": {"multi_select": [{"name": ticker[:100]} for ticker in tickers]},
        "Relevance": {"number": avg_relevance},
        "Tags": {"multi_select": [{"name": tag[:100]} for tag in tags]},
        "Status": {"select": {"name": "New"}},
        "Source URL": {"url": source_url or None},
        "Video ID": {"rich_text": to_rich_text(video_id)},
    }


def frame_blocks(frame: dict[str, Any]) -> list[dict[str, Any]]:
    category = normalize_text(frame.get("category"), "UNKNOWN")
    description = normalize_text(frame.get("description"), "No description")
    timestamp = int(frame.get("timestamp") or 0)
    relevance = normalize_text(frame.get("relevance"), "n/a")
    key_data = ", ".join(normalize_list(frame.get("key_data", []))) or "n/a"
    verbal_context = normalize_text(frame.get("verbal_context"), "n/a")
    insight = normalize_text(frame.get("insight"), "n/a")
    content_angle = normalize_text(frame.get("content_angle"), "n/a")
    tags = ", ".join(normalize_list(frame.get("tags", []))) or "n/a"
    source_url = normalize_text(frame.get("sourceUrl"), "")

    lines = [
        f"Relevance: {relevance}/5",
        f"What: {description}",
        f"Key Data: {key_data}",
        f"Presenter Says: {verbal_context}",
        f"Insight: {insight}",
        f"Content Angle: {content_angle}",
        f"Tags: {tags}",
    ]
    if source_url:
        lines.append(f"Watch: {source_url}")

    blocks: list[dict[str, Any]] = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": to_rich_text(f"[{category}] {description} ({mmss(timestamp)})"),
            },
        }
    ]

    for line in lines:
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": to_rich_text(line)},
            }
        )
    return blocks


def safe_call(action: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Notion %s failed: %s", action, exc)
        raise


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.database_id:
        raise RuntimeError("--database-id or NOTION_CONTENT_VAULT_DB is required")
    if not args.token:
        raise RuntimeError("--token or NOTION_TOKEN is required")

    annotations_path = resolve_path(args.annotations)
    annotations = load_json(annotations_path, default=[])
    if not isinstance(annotations, list):
        raise ValueError(f"Annotations must be a list: {annotations_path}")

    grouped = group_kept_annotations(annotations)
    if not grouped:
        LOGGER.info("No kept annotations to sync")
        print("Synced 0 videos to Notion")
        return 0

    notion = Client(auth=args.token)

    synced = 0
    failed = 0

    for video_id, frames in grouped.items():
        try:
            properties = build_page_properties(frames)
            page = safe_call(
                "page create",
                notion.pages.create,
                parent={"database_id": args.database_id},
                properties=properties,
            )
            time.sleep(args.delay)

            blocks: list[dict[str, Any]] = []
            for frame in sorted(frames, key=lambda item: int(item.get("timestamp") or 0)):
                blocks.extend(frame_blocks(frame))

            if blocks:
                safe_call(
                    "children append",
                    notion.blocks.children.append,
                    block_id=page["id"],
                    children=blocks[:100],
                )
                # Notion allows 100 blocks per append call.
                for start in range(100, len(blocks), 100):
                    time.sleep(args.delay)
                    safe_call(
                        "children append",
                        notion.blocks.children.append,
                        block_id=page["id"],
                        children=blocks[start : start + 100],
                    )
            time.sleep(args.delay)
            synced += 1
            LOGGER.info("Synced video %s with %s frames", video_id, len(frames))
        except Exception:
            failed += 1
            continue

    LOGGER.info("Notion sync complete (synced=%s failed=%s)", synced, failed)
    print(f"Synced {synced} videos to Notion (failed {failed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
