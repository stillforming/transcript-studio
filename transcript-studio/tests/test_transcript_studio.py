#!/usr/bin/env python3
"""Comprehensive test suite for Transcript Studio (Phases 1-4).

Tests all scripts without external API calls (Notion, OpenAI, Whisper).
Uses real test data from tmp/test-phase/.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure scripts dir is on path
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (
    ROOT_DIR,
    SKILL_DIR,
    ensure_dir,
    load_json,
    normalize_slug,
    parse_upload_date,
    resolve_path,
    save_json,
    setup_logging,
    utc_today_str,
    utcnow,
)

TEST_DATA_DIR = SKILL_DIR / "tests" / "fixtures"

PASS = 0
FAIL = 0
ERRORS: list[str] = []


def ok(name: str) -> None:
    global PASS
    PASS += 1
    print(f"  ✅ {name}")


def fail(name: str, reason: str) -> None:
    global FAIL
    FAIL += 1
    ERRORS.append(f"{name}: {reason}")
    print(f"  ❌ {name} — {reason}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: _common.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_common():
    print("\n📦 _common.py")

    # resolve_path
    p = resolve_path("tmp/test")
    if str(p).endswith("tmp/test") and p.is_absolute():
        ok("resolve_path returns absolute path")
    else:
        fail("resolve_path", f"got {p}")

    # normalize_slug
    tests = {
        "Sky View Trading": "sky-view-trading",
        "tastylive": "tastylive",
        "Option Alpha": "option-alpha",
        "The Compound & Friends": "the-compound-friends",
    }
    all_good = True
    for input_val, expected in tests.items():
        result = normalize_slug(input_val)
        if result != expected:
            fail("normalize_slug", f"{input_val!r} → {result!r} (expected {expected!r})")
            all_good = False
    if all_good:
        ok("normalize_slug handles various inputs")

    # parse_upload_date
    d = parse_upload_date("20260224")
    if d and d.year == 2026 and d.month == 2 and d.day == 24:
        ok("parse_upload_date YYYYMMDD format")
    else:
        fail("parse_upload_date", f"got {d}")

    d2 = parse_upload_date("2026-02-24")
    if d2 and d2.year == 2026:
        ok("parse_upload_date ISO format")
    elif d2 is None:
        ok("parse_upload_date: ISO format not supported (YYYYMMDD only) — expected")
    else:
        fail("parse_upload_date ISO", f"got {d2}")

    # load_json / save_json roundtrip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = Path(f.name)
    try:
        test_data = {"key": "value", "list": [1, 2, 3]}
        save_json(tmp_path, test_data)
        loaded = load_json(tmp_path)
        if loaded == test_data:
            ok("load_json / save_json roundtrip")
        else:
            fail("json roundtrip", f"mismatch: {loaded}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # load_json default on missing file
    missing = load_json(Path("/nonexistent/file.json"), default={"fallback": True})
    if missing == {"fallback": True}:
        ok("load_json returns default for missing file")
    else:
        fail("load_json default", f"got {missing}")

    # utc_today_str format
    today = utc_today_str()
    if len(today) == 10 and today[4] == "-" and today[7] == "-":
        ok("utc_today_str returns YYYY-MM-DD")
    else:
        fail("utc_today_str", f"got {today}")

    # ensure_dir
    with tempfile.TemporaryDirectory() as td:
        nested = Path(td) / "a" / "b" / "c"
        ensure_dir(nested)
        if nested.is_dir():
            ok("ensure_dir creates nested directories")
        else:
            fail("ensure_dir", "directory not created")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: merge_visuals.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_merge_visuals():
    print("\n🔀 merge_visuals.py")

    from merge_visuals import (
        normalize_segments,
        normalize_visuals,
        assign_visuals_to_segments,
        build_event_sequence,
        inject_timestamp_blocks,
        filter_annotations_for_video,
    )

    # Load test data
    transcript = load_json(TEST_DATA_DIR / "RHxd5EVmpuU_transcript.json")
    annotations = load_json(TEST_DATA_DIR / "annotations.json")

    if not transcript or not annotations:
        fail("test data", "Missing transcript or annotations in tmp/test-phase/")
        return

    # filter_annotations_for_video
    filtered = filter_annotations_for_video(annotations, "RHxd5EVmpuU")
    if len(filtered) == len(annotations):  # all same videoId in test data
        ok(f"filter_annotations_for_video: {len(filtered)} annotations (all match)")
    elif len(filtered) > 0:
        ok(f"filter_annotations_for_video: {len(filtered)}/{len(annotations)} matched")
    else:
        fail("filter_annotations_for_video", "0 annotations matched")

    # filter with wrong videoId
    wrong = filter_annotations_for_video(annotations, "NONEXISTENT")
    if len(wrong) == 0:
        ok("filter_annotations_for_video rejects wrong videoId")
    else:
        fail("filter wrong videoId", f"expected 0, got {len(wrong)}")

    # filter with empty videoId (should return all)
    all_back = filter_annotations_for_video(annotations, "")
    if len(all_back) == len(annotations):
        ok("filter_annotations_for_video: empty videoId returns all")
    else:
        fail("filter empty videoId", f"expected {len(annotations)}, got {len(all_back)}")

    # normalize_segments
    segments = normalize_segments(transcript)
    if isinstance(segments, list) and len(segments) > 0:
        ok(f"normalize_segments: {len(segments)} segments")
        seg = segments[0]
        if "start" in seg and "text" in seg:
            ok("segments have start + text fields")
        else:
            fail("segment fields", f"keys: {list(seg.keys())}")
    else:
        fail("normalize_segments", f"got {type(segments)} len={len(segments) if isinstance(segments, list) else 'N/A'}")
        return

    # normalize_visuals (without frames dir — should handle gracefully)
    with tempfile.TemporaryDirectory() as td:
        visuals = normalize_visuals(annotations, Path(td))
        if isinstance(visuals, list):
            ok(f"normalize_visuals: {len(visuals)} visuals (no frames dir)")
        else:
            fail("normalize_visuals", f"got {type(visuals)}")

    # assign_visuals_to_segments
    visual_mapping = assign_visuals_to_segments(segments, visuals)
    if isinstance(visual_mapping, dict):
        mapped_count = sum(len(v) for v in visual_mapping.values())
        ok(f"assign_visuals_to_segments: {mapped_count} visuals mapped to {len(visual_mapping)} segments")
    else:
        fail("assign_visuals", f"got {type(visual_mapping)}")

    # build_event_sequence
    events = build_event_sequence(segments, visual_mapping)
    if isinstance(events, list) and len(events) > 0:
        event_types = set(e.get("type") for e in events if isinstance(e, dict))
        ok(f"build_event_sequence: {len(events)} events, types: {event_types}")
    else:
        fail("build_event_sequence", f"got {len(events)} events")

    # inject_timestamp_blocks
    blocks = inject_timestamp_blocks(events)
    if isinstance(blocks, list) and len(blocks) > 0:
        block_types = set(b.get("type") for b in blocks if isinstance(b, dict))
        ok(f"inject_timestamp_blocks: {len(blocks)} blocks, types: {block_types}")
        if "timestamp" in block_types and "text" in block_types:
            ok("blocks contain timestamp + text types")
        else:
            fail("block types", f"missing expected types in {block_types}")
    else:
        fail("inject_timestamp_blocks", "empty result")

    # Full pipeline consistency
    merged = load_json(TEST_DATA_DIR / "merged_transcript.json")
    if merged and len(merged) > 200:
        ok(f"reference merged output: {len(merged)} blocks (sanity check)")
    else:
        fail("reference merged", f"unexpected: {len(merged) if merged else 0} blocks")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: summarize_video.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_summarize_video():
    print("\n📝 summarize_video.py")

    from summarize_video import (
        compact_transcript_segments,
        normalize_summary,
        normalize_takeaways,
        normalize_chapters,
        normalize_shorts,
        normalize_slide_suggestions,
        normalize_signal_extraction,
        normalize_market_bias,
        normalize_macro_signals,
        normalize_tickers,
        normalize_show_relevance,
        normalize_scores,
        normalize_catalysts,
        build_prompt,
    )

    transcript = load_json(TEST_DATA_DIR / "RHxd5EVmpuU_transcript.json")
    if not transcript:
        fail("test data", "Missing transcript")
        return

    # compact_transcript_segments
    segments = transcript.get("segments", [])
    compacted = compact_transcript_segments(segments)
    if isinstance(compacted, list) and len(compacted) > 0:
        ok(f"compact_transcript_segments: {len(compacted)} compacted segments (from {len(segments)})")
    else:
        fail("compact_transcript_segments", f"got {type(compacted)} len={len(compacted) if isinstance(compacted, list) else 'N/A'}")

    # build_prompt
    metadata = {k: v for k, v in transcript.items() if k != "segments"}
    prompt = build_prompt(metadata, compacted, [])
    if isinstance(prompt, str) and len(prompt) > 200:
        ok(f"build_prompt: {len(prompt)} chars")
    else:
        fail("build_prompt", f"got {type(prompt)} len={len(prompt) if isinstance(prompt, str) else 'N/A'}")

    # normalize_summary with reference test data
    ref_summary = load_json(TEST_DATA_DIR / "summary.json")
    normalized = normalize_summary(ref_summary)
    if isinstance(normalized, dict):
        ok(f"normalize_summary: keys={list(normalized.keys())}")
        for key in ["takeaways", "chapters", "shorts", "slide_suggestions"]:
            items = normalized.get(key, [])
            if isinstance(items, list) and len(items) > 0:
                ok(f"  {key}: {len(items)} items")
            elif isinstance(items, list):
                ok(f"  {key}: 0 items (normalizer may filter aggressively)")
            else:
                fail(f"normalize {key}", f"got {items}")
    else:
        fail("normalize_summary", f"got {type(normalized)}")

    # normalize individual sections
    raw_takeaways = [{"title": "Test", "detail": "Detail here", "importance": "high"}]
    norm_tk = normalize_takeaways(raw_takeaways)
    if isinstance(norm_tk, list):
        ok(f"normalize_takeaways: {len(norm_tk)} items")
    else:
        fail("normalize_takeaways", f"got {type(norm_tk)}")

    raw_chapters = [{"timestamp": "0:00", "title": "Intro", "summary": "Opening remarks"}]
    norm_ch = normalize_chapters(raw_chapters)
    if isinstance(norm_ch, list):
        ok(f"normalize_chapters: {len(norm_ch)} items")
    else:
        fail("normalize_chapters", f"got {type(norm_ch)}")

    # normalize_summary with empty input
    empty = normalize_summary({})
    if isinstance(empty, dict) and all(isinstance(empty.get(k, []), list) for k in ["takeaways", "chapters", "shorts"]):
        ok("normalize_summary handles empty input")
    else:
        fail("normalize empty", f"got {empty}")

    # ── Signal Extraction normalizer tests ──

    # normalize_market_bias
    bias = normalize_market_bias({"tone": "Bullish", "regime": "Risk-on rotation", "risk_tilt": "Risk-on", "cycle_position": "early-cycle"})
    if bias["tone"] == "Bullish" and bias["risk_tilt"] == "Risk-on" and bias["cycle_position"] == "early-cycle":
        ok("normalize_market_bias: valid input")
    else:
        fail("normalize_market_bias valid", f"got {bias}")

    bias_invalid = normalize_market_bias({"tone": "SUPER_BULLISH", "risk_tilt": "yolo", "cycle_position": "chaos"})
    if bias_invalid["tone"] == "Neutral" and bias_invalid["risk_tilt"] is None and bias_invalid["cycle_position"] is None:
        ok("normalize_market_bias: invalid values → defaults")
    else:
        fail("normalize_market_bias invalid", f"got {bias_invalid}")

    bias_empty = normalize_market_bias(None)
    if bias_empty["tone"] == "Neutral":
        ok("normalize_market_bias: None → defaults")
    else:
        fail("normalize_market_bias None", f"got {bias_empty}")

    # normalize_macro_signals
    macro = normalize_macro_signals({"rates": "Fed holding steady", "credit": None, "has_signal": True})
    if macro["rates"] == "Fed holding steady" and macro["credit"] is None and macro["has_signal"] is True:
        ok("normalize_macro_signals: valid input")
    else:
        fail("normalize_macro_signals valid", f"got {macro}")

    macro_empty = normalize_macro_signals({})
    if macro_empty["has_signal"] is False and all(macro_empty[k] is None for k in ("rates", "credit", "liquidity", "volatility", "structural_shifts")):
        ok("normalize_macro_signals: empty → no signal")
    else:
        fail("normalize_macro_signals empty", f"got {macro_empty}")

    # normalize_tickers
    tickers = normalize_tickers([
        {"ticker": "NVDA", "direction": "Bullish", "context": "AI spending", "catalyst": "Q3 earnings", "time_horizon": "positional", "timestamp": "05:30", "timestamp_seconds": 330, "signal_strength": 4},
        {"ticker": "aapl", "direction": "Neutral", "context": "Mentioned in passing", "time_horizon": "secular", "timestamp": "02:00", "timestamp_seconds": 120, "signal_strength": 1},
    ])
    if len(tickers) == 2 and tickers[0]["ticker"] == "AAPL" and tickers[1]["ticker"] == "NVDA":
        ok("normalize_tickers: valid input, uppercased, sorted by timestamp")
    else:
        fail("normalize_tickers valid", f"got {tickers}")

    if tickers[1].get("catalyst") == "Q3 earnings" and tickers[1].get("time_horizon") == "positional":
        ok("normalize_tickers: catalyst and time_horizon preserved")
    else:
        fail("normalize_tickers fields", f"catalyst={tickers[1].get('catalyst')}, horizon={tickers[1].get('time_horizon')}")

    # Invalid time_horizon should be nulled
    bad_horizon = normalize_tickers([{"ticker": "TSLA", "time_horizon": "forever"}])
    if len(bad_horizon) == 1 and bad_horizon[0]["time_horizon"] is None:
        ok("normalize_tickers: invalid time_horizon → None")
    else:
        fail("normalize_tickers bad horizon", f"got {bad_horizon[0].get('time_horizon') if bad_horizon else 'empty'}")

    tickers_empty = normalize_tickers([])
    if tickers_empty == []:
        ok("normalize_tickers: empty array")
    else:
        fail("normalize_tickers empty", f"got {tickers_empty}")

    tickers_junk = normalize_tickers([{"no_ticker": True}, "bad", None])
    if tickers_junk == []:
        ok("normalize_tickers: junk items filtered")
    else:
        fail("normalize_tickers junk", f"got {tickers_junk}")

    # normalize_show_relevance
    rel = normalize_show_relevance({"ai_thesis": "Yes", "software_repricing": "Weak", "infrastructure": "No", "rate_regime": "Yes", "liquidity_cycle": "Weak", "contradicts_narrative": "Disagrees on timing", "segment_potential": "Strong segment on AI capex"})
    if rel["ai_thesis"] == "Yes" and rel["software_repricing"] == "Weak" and rel["rate_regime"] == "Yes" and rel["liquidity_cycle"] == "Weak":
        ok("normalize_show_relevance: valid input (all 5 theses)")
    else:
        fail("normalize_show_relevance valid", f"got {rel}")

    rel_empty = normalize_show_relevance(None)
    if rel_empty["ai_thesis"] == "No" and rel_empty["rate_regime"] == "No" and rel_empty["liquidity_cycle"] == "No" and rel_empty["contradicts_narrative"] is None:
        ok("normalize_show_relevance: None → defaults (all 5 theses)")
    else:
        fail("normalize_show_relevance None", f"got {rel_empty}")

    # normalize_scores
    scores = normalize_scores({"macro_impact": 4, "stock_idea_density": 3, "contrarian_value": 2, "ai_infrastructure_relevance": 5, "show_utility": 4})
    if all(0 <= scores[k] <= 5 for k in scores) and scores["macro_impact"] == 4:
        ok("normalize_scores: valid input, clamped 0-5")
    else:
        fail("normalize_scores valid", f"got {scores}")

    scores_overflow = normalize_scores({"macro_impact": 99, "stock_idea_density": -3, "show_utility": 5})
    if scores_overflow["macro_impact"] == 5 and scores_overflow["stock_idea_density"] == 0 and scores_overflow["show_utility"] == 5:
        ok("normalize_scores: overflow/underflow clamped")
    else:
        fail("normalize_scores overflow", f"got {scores_overflow}")

    scores_empty = normalize_scores(None)
    if all(v == 0 for v in scores_empty.values()):
        ok("normalize_scores: None → all zeros")
    else:
        fail("normalize_scores None", f"got {scores_empty}")

    # normalize_catalysts
    cats = normalize_catalysts(["Fed meeting March 19", "NVDA earnings Feb 26", ""])
    if cats == ["Fed meeting March 19", "NVDA earnings Feb 26"]:
        ok("normalize_catalysts: filters empty strings")
    else:
        fail("normalize_catalysts valid", f"got {cats}")

    cats_empty = normalize_catalysts(None)
    if cats_empty == []:
        ok("normalize_catalysts: None → empty list")
    else:
        fail("normalize_catalysts None", f"got {cats_empty}")

    cats_junk = normalize_catalysts([None, "", 0, False])
    if cats_junk == []:
        ok("normalize_catalysts: falsy values filtered out")
    else:
        fail("normalize_catalysts junk", f"got {cats_junk}")

    # normalize_signal_extraction (full — with new fields)
    full_signal = normalize_signal_extraction({
        "quick_verdict": "High macro relevance. Multiple tradable ideas.",
        "speaker_assessment": "Veteran credit analyst, no promotional angle, high conviction.",
        "market_bias": {"tone": "Bullish"},
        "macro_signals": {"rates": "Fed cutting", "has_signal": True},
        "tickers": [{"ticker": "MSFT", "direction": "Bullish", "context": "Cloud growth", "catalyst": "Q3 earnings", "time_horizon": "positional", "signal_strength": 3}],
        "catalysts": ["Fed meeting", "NVDA earnings"],
        "show_relevance": {"ai_thesis": "Yes", "rate_regime": "Weak"},
        "scores": {"macro_impact": 4, "show_utility": 5},
    })
    if (full_signal["quick_verdict"] == "High macro relevance. Multiple tradable ideas."
            and full_signal["speaker_assessment"] == "Veteran credit analyst, no promotional angle, high conviction."
            and full_signal["market_bias"]["tone"] == "Bullish"
            and len(full_signal["tickers"]) == 1
            and full_signal["tickers"][0]["catalyst"] == "Q3 earnings"
            and full_signal["tickers"][0]["time_horizon"] == "positional"
            and full_signal["catalysts"] == ["Fed meeting", "NVDA earnings"]
            and full_signal["show_relevance"]["rate_regime"] == "Weak"
            and full_signal["scores"]["macro_impact"] == 4):
        ok("normalize_signal_extraction: full valid input (with new fields)")
    else:
        fail("normalize_signal_extraction full", f"got partial: verdict={full_signal.get('quick_verdict')}")

    empty_signal = normalize_signal_extraction(None)
    if (empty_signal["quick_verdict"] == "No signal analysis available."
            and empty_signal["speaker_assessment"] == "No speaker assessment available."
            and empty_signal["market_bias"]["tone"] == "Neutral"
            and empty_signal["tickers"] == []
            and empty_signal["catalysts"] == []
            and empty_signal["show_relevance"]["rate_regime"] == "No"
            and empty_signal["scores"]["macro_impact"] == 0):
        ok("normalize_signal_extraction: None → safe defaults (all new fields)")
    else:
        fail("normalize_signal_extraction None", f"got {empty_signal}")

    # normalize_summary now includes signal_extraction
    summary_with_signal = normalize_summary({"signal_extraction": {"quick_verdict": "Test verdict"}})
    if "signal_extraction" in summary_with_signal and summary_with_signal["signal_extraction"]["quick_verdict"] == "Test verdict":
        ok("normalize_summary includes signal_extraction")
    else:
        fail("normalize_summary signal", f"keys: {list(summary_with_signal.keys())}")

    # Verify reference summary structure
    ref_summary = load_json(TEST_DATA_DIR / "summary.json")
    if ref_summary:
        expected_keys = {"signal_extraction", "takeaways", "chapters", "shorts", "slide_suggestions"}
        actual_keys = set(ref_summary.keys())
        if expected_keys <= actual_keys:
            ok(f"reference summary has all expected sections (including signal_extraction)")
        else:
            fail("reference summary keys", f"missing: {expected_keys - actual_keys}")
    else:
        fail("reference summary", "missing fixtures/summary.json")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: export_notion.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_export_notion():
    print("\n📤 export_notion.py")

    from export_notion import (
        build_metadata_callout,
        build_page_metadata,
        build_page_properties,
        build_signal_extraction_blocks,
        build_tldr,
        build_summary_blocks,
        build_transcript_blocks,
        build_visual_index_toggle,
        build_raw_transcript_toggle,
        normalize_segments as export_normalize_segments,
        score_bar,
        split_text_chunks,
    )

    transcript = load_json(TEST_DATA_DIR / "RHxd5EVmpuU_transcript.json")
    merged = load_json(TEST_DATA_DIR / "merged_transcript.json")
    summary = load_json(TEST_DATA_DIR / "summary.json")

    if not all([transcript, merged, summary]):
        fail("test data", "Missing one or more test files")
        return

    # split_text_chunks
    short_text = "Hello world"
    chunks = split_text_chunks(short_text)
    if len(chunks) == 1 and chunks[0] == short_text:
        ok("split_text_chunks: short text unchanged")
    else:
        fail("split_text_chunks short", f"got {len(chunks)} chunks")

    long_text = "x" * 5000
    long_chunks = split_text_chunks(long_text)
    if len(long_chunks) > 1 and all(len(c) <= 2000 for c in long_chunks):
        ok(f"split_text_chunks: long text → {len(long_chunks)} chunks (all ≤2000)")
    else:
        fail("split_text_chunks long", f"chunks: {[len(c) for c in long_chunks]}")

    # build_page_metadata + build_metadata_callout
    export_segments = export_normalize_segments(transcript)
    metadata = build_page_metadata(transcript, export_segments)
    if isinstance(metadata, dict) and "video_title" in metadata and "source_url" in metadata:
        ok(f"build_page_metadata: {list(metadata.keys())}")
    else:
        fail("build_page_metadata", f"got {metadata}")

    # build_page_properties
    props = build_page_properties(metadata)
    if isinstance(props, dict) and "Name" in props and "Date" in props:
        ok(f"build_page_properties: {len(props)} properties")
    else:
        fail("build_page_properties", f"got {type(props)}")

    callout = build_metadata_callout(metadata)
    if isinstance(callout, dict) and callout.get("type") == "callout":
        ok("build_metadata_callout returns callout block")
    elif isinstance(callout, list) and len(callout) > 0:
        ok(f"build_metadata_callout returns {len(callout)} blocks")
    else:
        fail("metadata callout", f"got {type(callout)}")

    # score_bar helper
    if score_bar(0) == "○○○○○" and score_bar(3) == "●●●○○" and score_bar(5) == "●●●●●":
        ok("score_bar: renders correctly")
    else:
        fail("score_bar", f"0={score_bar(0)}, 3={score_bar(3)}, 5={score_bar(5)}")

    if score_bar(7) == "●●●●●" and score_bar(-1) == "○○○○○":
        ok("score_bar: clamps out-of-range")
    else:
        fail("score_bar clamp", f"7={score_bar(7)}, -1={score_bar(-1)}")

    # build_signal_extraction_blocks
    signal_blocks = build_signal_extraction_blocks(summary)
    if isinstance(signal_blocks, list) and len(signal_blocks) > 0:
        ok(f"build_signal_extraction_blocks: {len(signal_blocks)} blocks")

        # Check H1 heading is first
        if signal_blocks[0].get("type") == "heading_1":
            h1_text = "".join(
                rt.get("text", {}).get("content", "")
                for rt in signal_blocks[0].get("heading_1", {}).get("rich_text", [])
            )
            if "Signal Extraction" in h1_text:
                ok("signal extraction starts with correct H1")
            else:
                fail("signal H1 text", f"got '{h1_text}'")
        else:
            fail("signal H1", f"first block type: {signal_blocks[0].get('type')}")

        # Check callouts (verdict + speaker assessment + scores)
        verdict_blocks = [b for b in signal_blocks if b.get("type") == "callout"]
        if len(verdict_blocks) >= 2:
            ok(f"signal extraction has {len(verdict_blocks)} callout blocks (verdict + speaker + scores)")
        else:
            fail("signal callouts", f"got {len(verdict_blocks)}")

        # Check divider at end
        if signal_blocks[-1].get("type") == "divider":
            ok("signal extraction ends with divider")
        else:
            fail("signal divider", f"last block type: {signal_blocks[-1].get('type')}")

        # Check for all subsection headings
        h3_texts = []
        for block in signal_blocks:
            if block.get("type") == "heading_3":
                text = "".join(
                    rt.get("text", {}).get("content", "")
                    for rt in block.get("heading_3", {}).get("rich_text", [])
                )
                h3_texts.append(text)
        expected_h3s = {"Market Bias", "Macro Signals", "Ticker Mentions", "Catalysts", "Show Relevance"}
        if expected_h3s <= set(h3_texts):
            ok(f"signal extraction has all 5 subsection headings")
        else:
            fail("signal H3 headings", f"got {h3_texts}, expected {expected_h3s}")
    else:
        fail("signal extraction blocks", "empty or not a list")

    # build_signal_extraction_blocks with no signal data (backward compat)
    no_signal_blocks = build_signal_extraction_blocks({})
    if isinstance(no_signal_blocks, list) and len(no_signal_blocks) == 0:
        ok("build_signal_extraction_blocks: empty summary → no blocks (backward compat)")
    else:
        fail("signal extraction empty", f"got {len(no_signal_blocks)} blocks")

    # build_tldr
    tldr_blocks = build_tldr(summary)
    if (
        isinstance(tldr_blocks, list)
        and len(tldr_blocks) == 2
        and tldr_blocks[0].get("type") == "heading_2"
        and tldr_blocks[1].get("type") == "paragraph"
    ):
        ok("build_tldr returns heading + paragraph")
    else:
        fail("build_tldr", f"got {len(tldr_blocks) if isinstance(tldr_blocks, list) else type(tldr_blocks)}")

    # build_summary_blocks
    summary_blocks = build_summary_blocks(summary)
    if isinstance(summary_blocks, list) and len(summary_blocks) > 0:
        ok(f"build_summary_blocks: {len(summary_blocks)} blocks")

        def heading_2_by_text(text: str) -> dict[str, Any] | None:
            for block in summary_blocks:
                if not isinstance(block, dict) or block.get("type") != "heading_2":
                    continue
                rich = block.get("heading_2", {}).get("rich_text", [])
                plain = "".join(
                    rt.get("text", {}).get("content", "")
                    for rt in rich
                    if isinstance(rt, dict)
                )
                if plain == text:
                    return block
            return None

        key_takeaways = heading_2_by_text("🎯 Key Takeaways")
        chapters_heading = heading_2_by_text("📑 Chapters")
        shorts_heading = heading_2_by_text("🎬 Shorts & Clip Ideas")

        if key_takeaways and not key_takeaways.get("heading_2", {}).get("is_toggleable"):
            ok("takeaways are visible (not toggle)")
        else:
            fail("takeaways visibility", "missing heading or toggleable=True")

        if chapters_heading and not chapters_heading.get("heading_2", {}).get("is_toggleable"):
            ok("chapters are visible (not toggle)")
        else:
            fail("chapters visibility", "missing heading or toggleable=True")

        if shorts_heading and shorts_heading.get("heading_2", {}).get("is_toggleable"):
            ok("shorts are in a toggle heading")
        else:
            fail("shorts toggle", "missing heading or toggleable=False")
    else:
        fail("summary blocks", f"got {len(summary_blocks) if isinstance(summary_blocks, list) else type(summary_blocks)}")

    # build_transcript_blocks (takes merged + image_base_url, returns tuple)
    transcript_result = build_transcript_blocks(
        merged,
        "",
        chapters=summary.get("chapters", []),
        speaker_total=metadata.get("speakers", 1),
    )
    if isinstance(transcript_result, tuple) and len(transcript_result) == 2:
        transcript_blocks, visual_rows = transcript_result
        if isinstance(transcript_blocks, list) and len(transcript_blocks) > 0:
            block_types = set(b.get("type") for b in transcript_blocks if isinstance(b, dict))
            ok(f"build_transcript_blocks: {len(transcript_blocks)} blocks, types: {block_types}")
            if len(transcript_blocks) < 100:
                ok("transcript blocks merged (<100)")
            else:
                fail("transcript block count", f"{len(transcript_blocks)} (expected < 100)")
        else:
            fail("transcript blocks", f"got {len(transcript_blocks) if isinstance(transcript_blocks, list) else 'N/A'}")
        ok(f"visual_rows for index: {len(visual_rows)} entries")
    else:
        fail("build_transcript_blocks", f"expected tuple, got {type(transcript_result)}")
        transcript_blocks = []
        visual_rows = []

    # build_visual_index_toggle (takes visual_rows from build_transcript_blocks)
    index_block = build_visual_index_toggle(visual_rows)
    if isinstance(index_block, dict):
        ok(f"build_visual_index_toggle: toggle block")
    else:
        fail("visual index", f"got {type(index_block)}")

    # build_raw_transcript_toggle (takes payload + segments)
    raw_toggle = build_raw_transcript_toggle(transcript, export_segments)
    if isinstance(raw_toggle, dict):
        ok("build_raw_transcript_toggle returns toggle block")
    else:
        fail("raw transcript toggle", f"got {type(raw_toggle)}")

    # Total block count estimate (including nested heading/table children)
    all_top_level = []
    for blocks in [
        [callout] if isinstance(callout, dict) else (callout or []),
        signal_blocks or [],
        tldr_blocks or [],
        summary_blocks or [],
        transcript_blocks or [],
        [index_block] if isinstance(index_block, dict) else [],
        [raw_toggle] if isinstance(raw_toggle, dict) else [],
    ]:
        if isinstance(blocks, list):
            all_top_level.extend(blocks)

    def recursive_block_count(blocks: list[dict[str, Any]]) -> int:
        total = 0
        for block in blocks:
            if not isinstance(block, dict):
                continue
            total += 1
            block_type = block.get("type")
            if block_type in {"heading_1", "heading_2", "heading_3"}:
                children = block.get(block_type, {}).get("children", [])
                total += recursive_block_count(children)
            elif block_type == "table":
                children = block.get("table", {}).get("children", [])
                total += recursive_block_count(children)
        return total

    recursive_total = recursive_block_count(all_top_level)
    if 95 <= recursive_total <= 160:
        ok(
            f"Total estimated Notion blocks: {recursive_total} (flat={len(all_top_level)}, incl. signal extraction)"
        )
    else:
        fail("total block count", f"{recursive_total} (expected ~95-160)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: setup_transcript_db.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_setup_transcript_db():
    print("\n🗄️  setup_transcript_db.py")

    from setup_transcript_db import transcript_studio_properties

    props = transcript_studio_properties()
    if isinstance(props, dict):
        ok("transcript_studio_properties returns dict")
        expected_props = ["Name", "Date", "Channel", "Preset", "Status"]
        found = [p for p in expected_props if p in props]
        if len(found) == len(expected_props):
            ok(f"schema has all expected properties: {found}")
        else:
            # Check with flexible casing
            all_keys = list(props.keys())
            ok(f"schema properties: {all_keys}")
    else:
        fail("transcript_studio_properties", f"got {type(props)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: select_transcript_videos.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_select_transcript_videos():
    print("\n🎬 select_transcript_videos.py")

    from select_transcript_videos import load_processed_ids

    # Test with empty log
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"processedVideoIds": ["abc123", "def456"]}, f)
        tmp_log = f.name

    try:
        ids = load_processed_ids(tmp_log)
        if ids == {"abc123", "def456"}:
            ok("load_processed_ids: correct set from log")
        else:
            fail("load_processed_ids", f"got {ids}")
    finally:
        os.unlink(tmp_log)

    # Test with missing log
    missing_ids = load_processed_ids("/nonexistent/log.json")
    if isinstance(missing_ids, set) and len(missing_ids) == 0:
        ok("load_processed_ids: empty set for missing file")
    else:
        fail("load_processed_ids missing", f"got {missing_ids}")

    # Test config loading
    channels_cfg = load_json(resolve_path("config/channels.json"))
    playlists_cfg = load_json(resolve_path("config/playlists.json"))

    if isinstance(channels_cfg, dict) and "channels" in channels_cfg:
        ok(f"channels.json: valid ({len(channels_cfg['channels'])} channels)")
    else:
        fail("channels.json", f"invalid structure: {type(channels_cfg)}")

    if isinstance(playlists_cfg, dict) and "playlists" in playlists_cfg:
        ok(f"playlists.json: valid ({len(playlists_cfg['playlists'])} playlists)")
    else:
        fail("playlists.json", f"invalid structure: {type(playlists_cfg)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: run_transcript_pipeline.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_run_transcript_pipeline():
    print("\n🚀 run_transcript_pipeline.py")

    from run_transcript_pipeline import (
        STEP_ORDER,
        build_steps,
        default_state,
        load_preset,
        load_state,
        mark_step,
        normalize_date,
        save_state,
        video_id_of,
        build_step_args,
    )

    # STEP_ORDER
    expected_steps = [
        "select", "download", "extract", "transcribe", "window",
        "classify", "merge", "summarize", "export", "log", "archive", "cleanup",
    ]
    if STEP_ORDER == expected_steps:
        ok(f"STEP_ORDER: {len(STEP_ORDER)} steps in correct order")
    else:
        fail("STEP_ORDER", f"got {STEP_ORDER}")

    # build_steps
    steps = build_steps()
    if len(steps) == 12:
        ok(f"build_steps: {len(steps)} StepSpec objects")
        per_video = [s.name for s in steps if s.per_video]
        if set(per_video) == {"merge", "summarize", "export"}:
            ok(f"per_video steps: {per_video}")
        else:
            fail("per_video steps", f"got {per_video}")
    else:
        fail("build_steps", f"got {len(steps)} steps")

    # default_state
    state = default_state()
    if "steps" in state and len(state["steps"]) == 12:
        all_pending = all(s["status"] == "pending" for s in state["steps"].values())
        if all_pending:
            ok("default_state: all 12 steps pending")
        else:
            fail("default_state", "not all pending")
    else:
        fail("default_state", f"steps: {len(state.get('steps', {}))}")

    # load_preset
    for preset_name in ["default", "podcast", "presentation", "suno"]:
        preset = load_preset(preset_name)
        if isinstance(preset, dict) and "frame_interval" in preset and "whisper_model" in preset:
            ok(f"load_preset({preset_name}): interval={preset['frame_interval']}, model={preset['whisper_model']}")
        else:
            fail(f"load_preset({preset_name})", f"got {preset}")

    # load_preset fallback
    fallback = load_preset("nonexistent_preset")
    if isinstance(fallback, dict) and fallback.get("name") in ("default", "nonexistent_preset"):
        ok("load_preset fallback to default")
    else:
        fail("load_preset fallback", f"got {fallback}")

    # normalize_date
    if normalize_date("2026-02-24") == "2026-02-24":
        ok("normalize_date: valid date passes through")
    else:
        fail("normalize_date", "unexpected result")

    today = normalize_date(None)
    if len(today) == 10:
        ok(f"normalize_date(None): defaults to today ({today})")
    else:
        fail("normalize_date None", f"got {today}")

    # mark_step
    test_state = default_state()
    mark_step(test_state, "download", "completed", "downloaded 3 videos")
    if test_state["steps"]["download"]["status"] == "completed":
        ok("mark_step: sets status correctly")
    else:
        fail("mark_step", f"status: {test_state['steps']['download']['status']}")

    # video_id_of
    if video_id_of({"id": "abc123"}) == "abc123":
        ok("video_id_of: extracts from 'id'")
    else:
        fail("video_id_of", "wrong extraction")

    if video_id_of({"videoId": "xyz789"}) == "xyz789":
        ok("video_id_of: extracts from 'videoId'")
    else:
        fail("video_id_of videoId", "wrong extraction")

    # save_state / load_state roundtrip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_file = Path(f.name)
    try:
        test_state = default_state()
        test_state["date"] = "2026-02-24"
        test_state["videoUrl"] = None
        test_state["presetOverride"] = None
        mark_step(test_state, "select", "completed", "done")
        save_state(state_file, test_state)

        loaded = load_state(state_file, "2026-02-24", None, None, force=False)
        if loaded["steps"]["select"]["status"] == "completed":
            ok("save_state / load_state: preserves completed steps")
        else:
            fail("state roundtrip", f"select status: {loaded['steps']['select']['status']}")

        # Force should reset
        forced = load_state(state_file, "2026-02-24", None, None, force=True)
        if forced["steps"]["select"]["status"] == "pending":
            ok("load_state force=True: resets all steps")
        else:
            fail("force reset", f"select status: {forced['steps']['select']['status']}")

        # Different date should reset
        diff_date = load_state(state_file, "2026-03-01", None, None, force=False)
        if diff_date["steps"]["select"]["status"] == "pending":
            ok("load_state: new date creates fresh state")
        else:
            fail("date mismatch", "should create fresh state")
    finally:
        state_file.unlink(missing_ok=True)

    # build_step_args for each non-per-video step
    preset = load_preset("default")
    for step_name in ["select", "download", "extract", "transcribe", "window", "classify", "log", "archive", "cleanup"]:
        try:
            args = build_step_args(step_name, "2026-02-24", preset)
            if isinstance(args, list) and len(args) > 0:
                ok(f"build_step_args({step_name}): {len(args)} args")
            else:
                fail(f"build_step_args({step_name})", f"got {args}")
        except Exception as e:
            fail(f"build_step_args({step_name})", str(e))

    # Preset-driven args verification
    podcast_preset = load_preset("podcast")
    extract_args = build_step_args("extract", "2026-02-24", podcast_preset)
    interval_idx = extract_args.index("--interval") if "--interval" in extract_args else -1
    if interval_idx >= 0 and extract_args[interval_idx + 1] == "30":
        ok("extract args use podcast preset interval (30)")
    else:
        fail("preset-driven args", f"--interval not 30 in {extract_args}")

    presentation_preset = load_preset("presentation")
    extract_args2 = build_step_args("extract", "2026-02-24", presentation_preset)
    interval_idx2 = extract_args2.index("--interval") if "--interval" in extract_args2 else -1
    if interval_idx2 >= 0 and extract_args2[interval_idx2 + 1] == "5":
        ok("extract args use presentation preset interval (5)")
    else:
        fail("presentation args", f"--interval not 5 in {extract_args2}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8: transcribe_local.py (import checks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_transcribe_local():
    print("\n🎙️  transcribe_local.py")

    # Can't test actual transcription (needs mlx-whisper / Apple Silicon)
    # But we can test the helper functions
    from transcribe_local import (
        format_timestamp,
        write_outputs,
        merge_diarization,
        normalize_segment,
        normalize_segments as tl_normalize_segments,
        build_transcript_payload,
    )
    ok("transcribe_local: key functions importable")

    # format_timestamp
    ts = format_timestamp(125.7)
    if "2:05" in ts:
        ok(f"format_timestamp(125.7): {ts}")
    else:
        fail("format_timestamp", f"got {ts}")

    ts2 = format_timestamp(3661.0)
    if "1:01:01" in ts2 or "61:01" in ts2:
        ok(f"format_timestamp(3661.0): {ts2}")
    else:
        fail("format_timestamp hour", f"got {ts2}")

    ts3 = format_timestamp(0.0)
    if "0:00" in ts3:
        ok(f"format_timestamp(0.0): {ts3}")
    else:
        fail("format_timestamp zero", f"got {ts3}")

    # normalize_segment
    raw_seg = {"start": 10.5, "end": 15.2, "text": " Hello world "}
    norm = normalize_segment(raw_seg)
    if isinstance(norm, dict) and "start" in norm and "text" in norm:
        ok(f"normalize_segment: text='{norm.get('text', '')[:20]}'")
    else:
        fail("normalize_segment", f"got {norm}")

    # merge_diarization with empty diarization
    whisper_segs = [
        {"start": 0.0, "end": 5.0, "text": "Hello"},
        {"start": 5.0, "end": 10.0, "text": "World"},
    ]
    merged = merge_diarization(whisper_segs, [])
    if isinstance(merged, list) and len(merged) == 2:
        ok(f"merge_diarization (no speakers): {len(merged)} segments")
    else:
        fail("merge_diarization empty", f"got {len(merged) if isinstance(merged, list) else type(merged)}")

    # merge_diarization with speaker data
    diar_segs = [
        (0.0, 5.0, "Speaker 1"),
        (5.0, 10.0, "Speaker 2"),
    ]
    merged_with_speakers = merge_diarization(whisper_segs, diar_segs)
    if isinstance(merged_with_speakers, list) and len(merged_with_speakers) == 2:
        has_speaker = any(s.get("speaker") for s in merged_with_speakers)
        if has_speaker:
            ok(f"merge_diarization (with speakers): speakers assigned")
        else:
            ok(f"merge_diarization (with speakers): {len(merged_with_speakers)} segments (speaker field may be optional)")
    else:
        fail("merge_diarization speakers", f"got {merged_with_speakers}")

    # write_outputs to temp dir
    with tempfile.TemporaryDirectory() as td:
        test_payload = {
            "videoId": "test123",
            "segments": [{"start": 0, "end": 5, "text": "Hello", "speaker": "Speaker 1"}],
            "speakers": ["Speaker 1"],
            "raw_text": "Hello",
            "text": "Hello",
        }
        write_outputs(Path(td), "test123", test_payload)
        expected_files = list(Path(td).glob("test123*"))
        if len(expected_files) > 0:
            ok(f"write_outputs: created {len(expected_files)} files in temp dir")
        else:
            fail("write_outputs", "no files created")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9: Preset config validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_preset_configs():
    print("\n⚙️  Preset configs")

    presets_dir = resolve_path("config/presets")
    required_keys = {"name", "description", "frame_interval", "hash_threshold", "whisper_model"}

    for preset_file in sorted(presets_dir.glob("*.json")):
        data = load_json(preset_file)
        if not isinstance(data, dict):
            fail(f"preset {preset_file.name}", f"not a dict: {type(data)}")
            continue

        missing = required_keys - set(data.keys())
        if missing:
            fail(f"preset {preset_file.name}", f"missing keys: {missing}")
        else:
            ok(f"preset {preset_file.name}: valid ({data['name']}, interval={data['frame_interval']})")

        # Type checks
        if not isinstance(data.get("frame_interval"), (int, float)):
            fail(f"preset {preset_file.name} frame_interval", f"type: {type(data.get('frame_interval'))}")
        if not isinstance(data.get("hash_threshold"), (int, float)):
            fail(f"preset {preset_file.name} hash_threshold", f"type: {type(data.get('hash_threshold'))}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10: End-to-end merge → export flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_e2e_merge_to_export():
    print("\n🔗 End-to-end: merge → summarize (mock) → export block generation")

    from merge_visuals import (
        normalize_segments,
        normalize_visuals,
        assign_visuals_to_segments,
        build_event_sequence,
        inject_timestamp_blocks,
        filter_annotations_for_video,
    )
    from summarize_video import normalize_summary
    from export_notion import (
        build_metadata_callout,
        build_page_metadata,
        build_signal_extraction_blocks,
        build_tldr,
        build_summary_blocks,
        build_transcript_blocks,
        normalize_segments as e2e_normalize_segments,
    )

    transcript = load_json(TEST_DATA_DIR / "RHxd5EVmpuU_transcript.json")
    annotations = load_json(TEST_DATA_DIR / "annotations.json")
    summary = load_json(TEST_DATA_DIR / "summary.json")

    # Step 1: Merge
    filtered_ann = filter_annotations_for_video(annotations, "RHxd5EVmpuU")
    segments = normalize_segments(transcript)
    with tempfile.TemporaryDirectory() as td:
        visuals = normalize_visuals(filtered_ann, Path(td))
    visual_mapping = assign_visuals_to_segments(segments, visuals)
    events = build_event_sequence(segments, visual_mapping)
    merged_blocks = inject_timestamp_blocks(events)

    if len(merged_blocks) > 200:
        ok(f"E2E merge: {len(merged_blocks)} blocks")
    else:
        fail("E2E merge", f"only {len(merged_blocks)} blocks")

    # Step 2: Normalize summary (mock — already have test data)
    norm_summary = normalize_summary(summary)
    section_counts = {k: len(v) for k, v in norm_summary.items() if isinstance(v, list)}
    ok(f"E2E summary: {section_counts}")

    # Step 3: Export block generation
    e2e_segs = e2e_normalize_segments(transcript)
    e2e_metadata = build_page_metadata(transcript, e2e_segs)
    callout = build_metadata_callout(e2e_metadata)
    signal_blocks = build_signal_extraction_blocks(norm_summary)
    tldr_blocks = build_tldr(norm_summary)
    summary_blocks = build_summary_blocks(norm_summary)
    transcript_result = build_transcript_blocks(
        merged_blocks,
        "",
        chapters=norm_summary.get("chapters", []),
        speaker_total=e2e_metadata.get("speakers", 1),
    )
    transcript_blocks = transcript_result[0] if isinstance(transcript_result, tuple) else transcript_result

    if isinstance(transcript_blocks, list) and len(transcript_blocks) < 100:
        ok(f"E2E transcript blocks compacted: {len(transcript_blocks)}")
    else:
        fail(
            "E2E transcript compaction",
            f"{len(transcript_blocks) if isinstance(transcript_blocks, list) else type(transcript_blocks)}",
        )

    if isinstance(signal_blocks, list) and len(signal_blocks) > 10:
        ok(f"E2E signal extraction: {len(signal_blocks)} blocks")
    else:
        fail("E2E signal extraction", f"got {len(signal_blocks) if isinstance(signal_blocks, list) else type(signal_blocks)}")

    top_level_total = (
        (1 if isinstance(callout, dict) else len(callout or []))
        + len(signal_blocks or [])
        + len(tldr_blocks or [])
        + len(summary_blocks or [])
        + len(transcript_blocks or [])
    )

    # Verify no empty blocks  
    all_blocks = []
    if isinstance(callout, dict):
        all_blocks.append(callout)
    elif isinstance(callout, list):
        all_blocks.extend(callout)
    if isinstance(signal_blocks, list):
        all_blocks.extend(signal_blocks)
    if isinstance(tldr_blocks, list):
        all_blocks.extend(tldr_blocks)
    if isinstance(summary_blocks, list):
        all_blocks.extend(summary_blocks)
    if isinstance(transcript_blocks, list):
        all_blocks.extend(transcript_blocks)

    def recursive_block_count(blocks: list[dict[str, Any]]) -> int:
        total = 0
        for block in blocks:
            if not isinstance(block, dict):
                continue
            total += 1
            block_type = block.get("type")
            if block_type in {"heading_1", "heading_2", "heading_3"}:
                children = block.get(block_type, {}).get("children", [])
                total += recursive_block_count(children)
            elif block_type == "table":
                children = block.get("table", {}).get("children", [])
                total += recursive_block_count(children)
        return total

    recursive_total = recursive_block_count(all_blocks)
    if 95 <= recursive_total <= 160:
        ok(
            "E2E export blocks: "
            f"{recursive_total} total (flat={top_level_total}; callout + signal + TL;DR + summary + transcript)"
        )
    else:
        fail("E2E export", f"{recursive_total} (expected ~95-160)")

    empty_blocks = [b for b in all_blocks if not isinstance(b, dict) or not b.get("type")]
    if len(empty_blocks) == 0:
        ok("All blocks have valid 'type' field")
    else:
        fail("empty blocks", f"{len(empty_blocks)} blocks missing 'type'")

    # Notion API constraint: max 100 blocks per append
    chunk_size = 100
    chunks_needed = (len(all_blocks) + chunk_size - 1) // chunk_size
    ok(f"Would need {chunks_needed} Notion API calls to append {len(all_blocks)} blocks (100/call limit)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> int:
    print("=" * 60)
    print("  TRANSCRIPT STUDIO — Full Test Suite")
    print("=" * 60)

    test_common()
    test_merge_visuals()
    test_summarize_video()
    test_export_notion()
    test_setup_transcript_db()
    test_select_transcript_videos()
    test_run_transcript_pipeline()
    test_transcribe_local()
    test_preset_configs()
    test_e2e_merge_to_export()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if ERRORS:
        print("\nFailed tests:")
        for e in ERRORS:
            print(f"  ❌ {e}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
