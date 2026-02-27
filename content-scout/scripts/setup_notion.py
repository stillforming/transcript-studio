#!/usr/bin/env python3
"""Create Notion databases required by Content Scout."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from notion_client import Client

from _common import setup_logging

LOGGER = logging.getLogger("content_scout.setup_notion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=os.environ.get("NOTION_TOKEN", ""), help="Notion API token")
    parser.add_argument("--parent-page-id", required=True, help="Parent Notion page ID")
    parser.add_argument(
        "--output",
        default="tmp/notion_database_ids.json",
        help="Optional path to save created database IDs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def create_database(
    notion: Client,
    parent_page_id: str,
    title: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    LOGGER.info("Creating Notion database: %s", title)
    return notion.databases.create(
        parent={"type": "page_id", "page_id": parent_page_id},
        title=[{"type": "text", "text": {"content": title}}],
        properties=properties,
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.token:
        raise RuntimeError("--token or NOTION_TOKEN is required")

    notion = Client(auth=args.token)

    content_vault = create_database(
        notion,
        args.parent_page_id,
        "Content Vault",
        {
            "Name": {"title": {}},
            "Date": {"date": {}},
            "Channel": {"select": {}},
            "Tickers": {"multi_select": {}},
            "Relevance": {"number": {"format": "number"}},
            "Tags": {"multi_select": {}},
            "Status": {
                "select": {
                    "options": [
                        {"name": "New", "color": "blue"},
                        {"name": "In Review", "color": "yellow"},
                        {"name": "Published", "color": "green"},
                    ]
                }
            },
            "Source URL": {"url": {}},
            "Video ID": {"rich_text": {}},
        },
    )

    daily_briefs = create_database(
        notion,
        args.parent_page_id,
        "Daily Briefs",
        {
            "Name": {"title": {}},
            "Date": {"date": {}},
            "Summary": {"rich_text": {}},
            "Status": {
                "select": {
                    "options": [
                        {"name": "Draft", "color": "yellow"},
                        {"name": "Ready", "color": "green"},
                    ]
                }
            },
            "File Path": {"rich_text": {}},
        },
    )

    channels = create_database(
        notion,
        args.parent_page_id,
        "Channels",
        {
            "Name": {"title": {}},
            "Channel ID": {"rich_text": {}},
            "Slug": {"rich_text": {}},
            "URL": {"url": {}},
            "Priority": {
                "select": {
                    "options": [
                        {"name": "high", "color": "red"},
                        {"name": "medium", "color": "yellow"},
                        {"name": "low", "color": "gray"},
                    ]
                }
            },
            "Status": {
                "select": {
                    "options": [
                        {"name": "active", "color": "green"},
                        {"name": "paused", "color": "gray"},
                    ]
                }
            },
            "Last Seen": {"date": {}},
        },
    )

    ids = {
        "contentVaultDatabaseId": content_vault.get("id"),
        "dailyBriefsDatabaseId": daily_briefs.get("id"),
        "channelsDatabaseId": channels.get("id"),
    }

    output_path = args.output
    if output_path:
        from _common import resolve_path, save_json  # local import to keep startup cheap

        save_json(resolve_path(output_path), ids)

    print(json.dumps(ids, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
