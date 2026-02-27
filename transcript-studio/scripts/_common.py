#!/usr/bin/env python3
"""Shared utilities for Content Scout scripts."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slugify import slugify

import os as _os

# WORKDIR env var overrides default; falls back to current working directory.
# This keeps output (tmp/, content-vault/) out of the skill installation.
ROOT_DIR = Path(_os.environ.get("CONTENT_SCOUT_WORKDIR", "")).resolve() if _os.environ.get("CONTENT_SCOUT_WORKDIR") else Path.cwd()

# Where the skill's bundled files live (config/, scripts/)
SKILL_DIR = Path(__file__).resolve().parents[1]


class ContentScoutError(RuntimeError):
    """Raised when a deterministic pipeline step cannot continue."""


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(path_str: str) -> Path:
    """Resolve a relative path. Bundled assets (config/) resolve to SKILL_DIR;
    everything else (tmp/, content-vault/, output) resolves to ROOT_DIR (workdir)."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    # Bundled skill assets live in SKILL_DIR
    if path_str.startswith("config/"):
        return SKILL_DIR / path
    return ROOT_DIR / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def utcnow() -> datetime:
    return datetime.now(UTC)


def utc_today_str() -> str:
    return utcnow().date().isoformat()


def parse_upload_date(value: str) -> datetime | None:
    """Parse yt-dlp upload_date (`YYYYMMDD`) into UTC datetime."""
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC)


def normalize_slug(value: str) -> str:
    return slugify(value) if value else ""
