#!/usr/bin/env python3
"""Create the Transcript Studio Notion database."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from notion_client import Client

from _common import resolve_path, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.setup_transcript_db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=os.environ.get("NOTION_TOKEN", ""), help="Notion API token")
    parser.add_argument("--parent-page-id", required=True, help="Parent Notion page ID")
    parser.add_argument("--output", default="", help="Optional path to save created database ID JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def transcript_studio_properties() -> dict[str, Any]:
    return {
        "Name": {"title": {}},
        "Date": {"date": {}},
        "Channel": {"select": {}},
        "Preset": {
            "select": {
                "options": [
                    {"name": "Default", "color": "gray"},
                    {"name": "Podcast", "color": "blue"},
                    {"name": "Presentation", "color": "green"},
                    {"name": "Suno", "color": "orange"},
                ]
            }
        },
        "Duration": {"number": {"format": "number"}},
        "Speakers": {"number": {"format": "number"}},
        "Status": {
            "select": {
                "options": [
                    {"name": "Processing", "color": "yellow"},
                    {"name": "Ready", "color": "green"},
                    {"name": "Reviewed", "color": "blue"},
                    {"name": "Error", "color": "red"},
                ]
            }
        },
        "Source URL": {"url": {}},
        "Video ID": {"rich_text": {}},
        "Tags": {"multi_select": {}},
        "Playlist": {"select": {}},
    }


def create_database(notion: Client, parent_page_id: str) -> dict[str, Any]:
    LOGGER.info("Creating Notion database: Transcript Studio")
    return notion.databases.create(
        parent={"type": "page_id", "page_id": parent_page_id},
        title=[{"type": "text", "text": {"content": "Transcript Studio"}}],
        properties=transcript_studio_properties(),
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        if not args.token:
            raise RuntimeError("--token or NOTION_TOKEN is required")

        notion = Client(auth=args.token)
        database = create_database(notion, args.parent_page_id)
        database_id = str(database.get("id") or "").strip()
        if not database_id:
            raise RuntimeError("Notion create returned no database id")

        if args.output:
            payload = {"transcriptStudioDatabaseId": database_id}
            save_json(resolve_path(args.output), payload)
            LOGGER.info("Saved Transcript Studio database id to %s", resolve_path(args.output))

        print(database_id)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("setup_transcript_db failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
