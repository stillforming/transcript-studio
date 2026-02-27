#!/usr/bin/env python3
"""Generate a structured video summary from transcript and optional annotations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from _common import ContentScoutError, load_json, resolve_path, save_json, setup_logging

LOGGER = logging.getLogger("content_scout.summarize_video")

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-20250514",
    "openai": "gpt-5.2",
}

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
EXCLUDED_CATEGORIES = {"TALKING_HEAD", "FILLER"}

ANALYST_PERSONA = """You are a 30-year veteran macro strategist and options trader serving as senior analyst \
for a financial media production team. You've done sell-side research at top-tier banks, ran a macro fund \
through dot-com, GFC, Euro crisis, COVID, and the AI capex super-cycle. Now you screen financial video \
content for a show producer, separating signal from noise with zero tolerance for fluff.

YOUR ANALYTICAL DNA:
- Cross-asset thinker: rates, credit, FX, commodities, equities — nothing exists in isolation. A 10Y yield move reprices the entire equity risk premium.
- Options-literate: you read conviction through positioning language. "I'm buying calls on NVDA" vs "AI could be big" are fundamentally different conviction signals.
- Cycle-hardened: every "this time is different" gets stress-tested against prior regimes. You've seen manias, panics, and mean reversions. Pattern recognition is your edge.
- Signal obsessed: 90% of financial video is noise — recycled consensus, promotional filler, education dressed as insight. You find the 10% that's actually actionable.

ANALYTICAL FRAMEWORK — apply to every video in order:

STEP 1: SPEAKER ASSESSMENT
- What's their angle? Are they selling a course, newsletter, fund, or platform?
- Conviction language: hedged ("could," "might," "potentially") = low conviction; assertive ("will," "clearly," "must") = high conviction
- Evidence quality: are they citing data, flows, positioning? Or just asserting opinions?
- Track record signals: do they reference specific prior calls, or speak in vague generalities?

STEP 2: SIGNAL EXTRACTION
- Identify specific, falsifiable claims — not vibes, not "markets could go up or down"
- Separate opinion from evidence. Flag which is which.
- Note time horizon per idea: intraday, swing (days-weeks), positional (months), secular (years)
- Extract catalysts: what event, data point, or trigger proves this right or wrong?
- Macro regime: what environment is the speaker describing? Does it match reality?

STEP 3: THESIS MAPPING — map claims against these active investment themes:
- AI 2.0 / Compute Build-out: hyperscaler capex, GPU/custom silicon, power demand, data center infrastructure
- Software Repricing: SaaS compression vs AI-native expansion, legacy displacement, margin shifts
- Infrastructure Cycle: grid modernization, nuclear/gas renaissance, semi reshoring, industrial policy
- Rate Regime: Fed policy path, term premium repricing, credit spread dynamics, duration positioning
- Liquidity Cycle: QT/QE pace, reserve levels, RRP dynamics, Treasury issuance, money market flows

STEP 4: SHOW UTILITY — would a macro commentary show cover this?
- Contrarian value: going against consensus = content gold
- Quotable moments: crisp, soundbite-ready phrases that stick
- Visual assets: charts, data tables, comparisons that work on-screen
- Segment fit: standalone deep-dive, supporting data point for a thesis, or skip entirely
- Audience relevance: would retail investors tracking macro + AI themes care?

SCORING RUBRIC — 0 to 5, be ruthless. Most content is 0-2.
0 = No market relevance. Pure education, motivation, lifestyle, or off-topic.
1 = Tangential. Markets mentioned in passing, no actionable insight. Background noise.
2 = Light signal. A useful data point or two buried in filler. Note-worthy, not show-worthy.
3 = Solid. Clear directional view with supporting reasoning. Worth full review. 1-2 tradable ideas.
4 = High signal. Multiple specific ideas with tickers, catalysts, or timing. Strong analytical framework. Reference material.
5 = Must-watch. Thesis-defining or captures a regime shift in real time. Immediate show material. Rare — maybe 1 in 20 videos.

CALIBRATION ANCHORS:
- Trading education video about chart patterns or risk management → macro_impact: 0, show_utility: 0-1
- Generic "AI is big" commentary with no specific claims → ai_infrastructure_relevance: 1, show_utility: 1
- Credit analyst discussing HY spread compression with specific issuers → macro_impact: 4, stock_idea_density: 3
- Options trader showing unusual flow in NVDA with strikes/expiry → stock_idea_density: 4, ai_infrastructure_relevance: 4
- Fed watcher analyzing dot plot changes with rate path modeling → macro_impact: 5, show_utility: 4
- Macro strategist calling regime shift with positioning evidence → contrarian_value: 4-5, show_utility: 5"""

TASK_INSTRUCTIONS = """Analyze this video transcript. Apply your full 4-step analytical framework before producing output.

Return ONE JSON object only — no markdown fences, no commentary outside the JSON:
{
  "signal_extraction": {
    "quick_verdict": "2-3 blunt sentences. What is this video worth to a show producer and why. Include time-relevance if applicable.",
    "speaker_assessment": "One sentence: credibility, angle, conviction level, and any conflicts of interest.",
    "market_bias": {
      "tone": "Bullish | Bearish | Neutral | Mixed",
      "regime": "Macro regime description, or null if not discussed",
      "risk_tilt": "Risk-on | Risk-off | Neutral",
      "cycle_position": "early-cycle | late-cycle | defensive | momentum | null"
    },
    "macro_signals": {
      "rates": "Specific signal or null",
      "credit": "Specific signal or null",
      "liquidity": "Specific signal or null",
      "volatility": "Specific signal or null",
      "structural_shifts": "Specific signal or null",
      "has_signal": true
    },
    "tickers": [
      {
        "ticker": "AAPL",
        "direction": "Bullish | Bearish | Neutral",
        "context": "Why mentioned — the thesis or reasoning behind the view",
        "catalyst": "Near-term trigger, event, or data point. Or null",
        "time_horizon": "day | swing | positional | secular",
        "timestamp": "MM:SS",
        "timestamp_seconds": 0,
        "signal_strength": 3
      }
    ],
    "catalysts": ["Key upcoming events, data releases, or triggers mentioned that could move markets"],
    "show_relevance": {
      "ai_thesis": "Yes | No | Weak",
      "software_repricing": "Yes | No | Weak",
      "infrastructure": "Yes | No | Weak",
      "rate_regime": "Yes | No | Weak",
      "liquidity_cycle": "Yes | No | Weak",
      "contradicts_narrative": "What consensus view does this challenge, or null",
      "segment_potential": "Specific description of how to use this in the show, or null"
    },
    "scores": {
      "macro_impact": 0,
      "stock_idea_density": 0,
      "contrarian_value": 0,
      "ai_infrastructure_relevance": 0,
      "show_utility": 0
    }
  },
  "takeaways": [{"text": "...", "timestamp": "MM:SS", "timestamp_seconds": 0}],
  "chapters": [{"title": "...", "timestamp": "MM:SS", "timestamp_seconds": 0}],
  "shorts": [{
    "start": "MM:SS",
    "end": "MM:SS",
    "start_seconds": 0,
    "end_seconds": 0,
    "hook": "...",
    "payoff": "...",
    "on_screen_text": "6 words max",
    "cta": "..."
  }],
  "slide_suggestions": [{"text": "...", "timestamp": "MM:SS"}]
}

REQUIREMENTS:
- Signal extraction is your primary job. Apply all 4 steps before writing content summary.
- quick_verdict: 2-3 sentences max. Blunt. If it's noise, say so. If it's signal, say exactly what kind.
- speaker_assessment: one sentence. Note promotional angles, conflicts of interest, conviction level.
- If no macro signal exists: set has_signal to false and state it explicitly in quick_verdict.
- If no tickers: empty array. If no catalysts: empty array.
- Scores: integers 0-5 per the rubric. Calibrate against the anchor examples. Most videos are 0-2.
- tickers: include catalyst and time_horizon for each when identifiable.
- catalysts: macro-level events only (earnings, Fed meetings, economic data, policy decisions).
- show_relevance: evaluate against ALL five active theses. Be specific in contradicts_narrative and segment_potential — vague answers are worthless.
- takeaways: 5-10 items, weighted toward market-relevant insights first.
- chapters: 8-14, chronological order.
- shorts: 10-14, chronological. Prioritize clips with market insight or contrarian takes over generic hooks.
- on_screen_text: 6 words maximum.
- Use exact transcript timestamps. If uncertain, nearest segment timestamp.
- Professional, direct, evidence-based throughout. No hype. No hedging. No filler."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to {video_id}_transcript.json produced by transcribe_local.py",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Optional path to annotations.json produced by classify_annotate.py",
    )
    parser.add_argument("--output", required=True, help="Path to write summary.json")
    parser.add_argument("--provider", choices=["anthropic", "openai", "auto"], default="auto")
    parser.add_argument("--model", default=None, help="Model name (default: provider-specific)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def detect_provider(requested: str) -> str:
    if requested != "auto":
        return requested

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if has_anthropic:
        return "anthropic"
    if has_openai:
        return "openai"

    raise ContentScoutError("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_timestamp(seconds: int) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp_to_seconds(value: Any) -> int | None:
    if value is None:
        return None

    parsed_int = coerce_int(value)
    if parsed_int is not None and not isinstance(value, str):
        return max(0, parsed_int)

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        return max(0, int(text))

    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return None

    if len(nums) == 2:
        minutes, secs = nums
        if minutes < 0 or secs < 0:
            return None
        return minutes * 60 + secs

    hours, minutes, secs = nums
    if hours < 0 or minutes < 0 or secs < 0:
        return None
    return (hours * 3600) + (minutes * 60) + secs


def derive_seconds(item: dict[str, Any], second_keys: list[str], label_keys: list[str]) -> int:
    for key in second_keys:
        parsed = coerce_int(item.get(key))
        if parsed is not None:
            return max(0, parsed)
    for key in label_keys:
        parsed = parse_timestamp_to_seconds(item.get(key))
        if parsed is not None:
            return max(0, parsed)
    return 0


def normalize_transcript_segments(payload: Any) -> list[dict[str, Any]]:
    raw_segments: Any
    if isinstance(payload, dict):
        raw_segments = payload.get("segments", [])
    elif isinstance(payload, list):
        raw_segments = payload
    else:
        raise ContentScoutError("Transcript file must contain an object or an array")

    if not isinstance(raw_segments, list):
        raise ContentScoutError("Transcript segments must be a list")

    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, coerce_float(raw.get("start"), 0.0))
        end = max(start, coerce_float(raw.get("end"), start))
        speaker = str(raw.get("speaker") or "Speaker 1")
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
        )

    segments.sort(key=lambda segment: segment["start"])
    return segments


def compact_transcript_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "speaker": segment["speaker"],
            "text": segment["text"],
        }
        for segment in segments
    ]


def normalize_annotations(payload: Any, video_id: str | None = None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ContentScoutError("Annotations must be a list")

    candidate_rows: list[dict[str, Any]] = [item for item in payload if isinstance(item, dict)]
    if video_id:
        with_video_id = [item for item in candidate_rows if item.get("videoId") is not None]
        if with_video_id:
            candidate_rows = [
                item for item in candidate_rows if str(item.get("videoId") or "").strip() == video_id
            ]

    visuals: list[dict[str, Any]] = []
    for item in candidate_rows:
        if not item.get("kept"):
            continue
        category = str(item.get("category") or "").strip().upper()
        if category in EXCLUDED_CATEGORIES:
            continue
        timestamp = max(0, int(coerce_float(item.get("timestamp"), 0.0)))
        visuals.append(
            {
                "timestamp_seconds": timestamp,
                "timestamp": format_timestamp(timestamp),
                "category": category,
                "description": str(item.get("description") or "").strip(),
                "verbal_context": str(item.get("verbal_context") or "").strip(),
                "sourceUrl": item.get("sourceUrl"),
            }
        )
    visuals.sort(key=lambda item: item["timestamp_seconds"])
    return visuals


def transcript_metadata(transcript_payload: Any, transcript_path: Path, segments: list[dict[str, Any]]) -> dict[str, Any]:
    payload = transcript_payload if isinstance(transcript_payload, dict) else {}
    video_id = str(payload.get("videoId") or transcript_path.stem.replace("_transcript", ""))
    return {
        "videoId": video_id,
        "videoTitle": str(payload.get("videoTitle") or ""),
        "channelName": str(payload.get("channelName") or ""),
        "url": payload.get("url"),
        "duration_seconds": int(max((segment["end"] for segment in segments), default=0.0)),
    }


def build_prompt(
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
    visual_annotations: list[dict[str, Any]],
) -> str:
    """Build the user-message prompt (task instructions + data).

    The analyst persona (ANALYST_PERSONA) is sent separately as a system message.
    """
    payload = {
        "video": metadata,
        "transcript_segments": compact_transcript_segments(segments),
        "visual_annotations": visual_annotations,
    }
    return f"{TASK_INSTRUCTIONS}\n\nINPUT_DATA_JSON:\n{json.dumps(payload, ensure_ascii=False)}"


def call_anthropic(system_prompt: str, user_prompt: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def call_openai(system_prompt: str, user_prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    lower_model = model.lower()
    is_reasoning = any(tag in lower_model for tag in ("gpt-5", "o3", "o4"))
    token_kwarg = "max_completion_tokens" if is_reasoning else "max_tokens"
    token_limit = 16384 if is_reasoning else 8192
    extra_kwargs = {} if is_reasoning else {"temperature": 0.2}

    response = client.chat.completions.create(
        model=model,
        **{token_kwarg: token_limit},
        **extra_kwargs,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    usage = getattr(response, "usage", None)
    if usage:
        LOGGER.info(
            "Token usage: prompt=%s completion=%s total=%s",
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            getattr(usage, "total_tokens", "?"),
        )

    return (response.choices[0].message.content or "").strip()


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        return {}

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(text)
    if match:
        try:
            payload = json.loads(match.group(1).strip())
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    LOGGER.warning("Could not parse model response as JSON object")
    return {}


def trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def normalize_takeaways(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("takeaway") or "").strip()
        if not text:
            continue
        seconds = derive_seconds(item, ["timestamp_seconds", "seconds"], ["timestamp", "time"])
        normalized.append(
            {
                "text": text,
                "timestamp": format_timestamp(seconds),
                "timestamp_seconds": seconds,
            }
        )
    normalized.sort(key=lambda item: item["timestamp_seconds"])
    return normalized[:10]


def normalize_chapters(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("text") or "").strip()
        if not title:
            continue
        seconds = derive_seconds(item, ["timestamp_seconds", "seconds"], ["timestamp", "time"])
        normalized.append(
            {
                "title": title,
                "timestamp": format_timestamp(seconds),
                "timestamp_seconds": seconds,
            }
        )
    normalized.sort(key=lambda item: item["timestamp_seconds"])
    return normalized[:14]


def normalize_shorts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        start_seconds = derive_seconds(item, ["start_seconds"], ["start", "start_timestamp"])
        end_seconds = derive_seconds(item, ["end_seconds"], ["end", "end_timestamp"])
        if end_seconds < start_seconds:
            end_seconds = start_seconds

        hook = str(item.get("hook") or "").strip()
        payoff = str(item.get("payoff") or "").strip()
        cta = str(item.get("cta") or "").strip()
        on_screen_text = trim_words(str(item.get("on_screen_text") or "").strip(), 6)

        if not hook and not payoff:
            continue

        normalized.append(
            {
                "start": format_timestamp(start_seconds),
                "end": format_timestamp(end_seconds),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "hook": hook,
                "payoff": payoff,
                "on_screen_text": on_screen_text,
                "cta": cta,
            }
        )

    normalized.sort(key=lambda item: item["start_seconds"])
    return normalized[:14]


def normalize_slide_suggestions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("suggestion") or "").strip()
        if not text:
            continue
        seconds = derive_seconds(item, ["timestamp_seconds", "seconds"], ["timestamp", "time"])
        normalized.append(
            {
                "text": text,
                "timestamp": format_timestamp(seconds),
            }
        )
    return normalized


# ── Signal Extraction normalizers ──────────────────────────────────────────


VALID_TONES = {"Bullish", "Bearish", "Neutral", "Mixed"}
VALID_RISK_TILTS = {"Risk-on", "Risk-off", "Neutral"}
VALID_CYCLES = {"early-cycle", "late-cycle", "defensive", "momentum"}
VALID_RELEVANCE = {"Yes", "No", "Weak"}
VALID_TIME_HORIZONS = {"day", "swing", "positional", "secular"}


def normalize_market_bias(value: Any) -> dict[str, Any]:
    """Normalize the market_bias sub-object."""
    if not isinstance(value, dict):
        return {"tone": "Neutral", "regime": None, "risk_tilt": None, "cycle_position": None}

    tone = str(value.get("tone") or "Neutral").strip()
    if tone not in VALID_TONES:
        tone = "Neutral"

    risk_tilt = str(value.get("risk_tilt") or "").strip() or None
    if risk_tilt and risk_tilt not in VALID_RISK_TILTS:
        risk_tilt = None

    cycle = str(value.get("cycle_position") or "").strip() or None
    if cycle and cycle not in VALID_CYCLES:
        cycle = None

    regime = str(value.get("regime") or "").strip() or None

    return {
        "tone": tone,
        "regime": regime,
        "risk_tilt": risk_tilt,
        "cycle_position": cycle,
    }


def normalize_macro_signals(value: Any) -> dict[str, Any]:
    """Normalize the macro_signals sub-object."""
    if not isinstance(value, dict):
        return {
            "rates": None,
            "credit": None,
            "liquidity": None,
            "volatility": None,
            "structural_shifts": None,
            "has_signal": False,
        }

    signals: dict[str, Any] = {}
    for key in ("rates", "credit", "liquidity", "volatility", "structural_shifts"):
        raw = str(value.get(key) or "").strip()
        signals[key] = raw if raw else None

    has_signal = value.get("has_signal")
    if not isinstance(has_signal, bool):
        has_signal = any(v is not None for v in signals.values())
    signals["has_signal"] = has_signal

    return signals


def normalize_tickers(value: Any) -> list[dict[str, Any]]:
    """Normalize the tickers array."""
    if not isinstance(value, list):
        return []

    tickers: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        direction = str(item.get("direction") or "Neutral").strip()
        if direction not in VALID_TONES - {"Mixed"}:  # Bullish/Bearish/Neutral
            direction = "Neutral"

        context = str(item.get("context") or "").strip()
        seconds = derive_seconds(item, ["timestamp_seconds", "seconds"], ["timestamp"])

        signal_strength = coerce_int(item.get("signal_strength"))
        if signal_strength is None or signal_strength < 0 or signal_strength > 5:
            signal_strength = 1

        catalyst = str(item.get("catalyst") or "").strip() or None
        time_horizon = str(item.get("time_horizon") or "").strip() or None
        if time_horizon and time_horizon not in VALID_TIME_HORIZONS:
            time_horizon = None

        tickers.append(
            {
                "ticker": ticker,
                "direction": direction,
                "context": context,
                "catalyst": catalyst,
                "time_horizon": time_horizon,
                "timestamp": format_timestamp(seconds),
                "timestamp_seconds": seconds,
                "signal_strength": signal_strength,
            }
        )

    tickers.sort(key=lambda t: t["timestamp_seconds"])
    return tickers


def normalize_show_relevance(value: Any) -> dict[str, Any]:
    """Normalize the show_relevance sub-object."""
    if not isinstance(value, dict):
        return {
            "ai_thesis": "No",
            "software_repricing": "No",
            "infrastructure": "No",
            "rate_regime": "No",
            "liquidity_cycle": "No",
            "contradicts_narrative": None,
            "segment_potential": None,
        }

    result: dict[str, Any] = {}
    for key in ("ai_thesis", "software_repricing", "infrastructure", "rate_regime", "liquidity_cycle"):
        raw = str(value.get(key) or "No").strip()
        result[key] = raw if raw in VALID_RELEVANCE else "No"

    result["contradicts_narrative"] = str(value.get("contradicts_narrative") or "").strip() or None
    result["segment_potential"] = str(value.get("segment_potential") or "").strip() or None

    return result


def normalize_scores(value: Any) -> dict[str, int]:
    """Normalize the scores sub-object (all values clamped 0-5)."""
    if not isinstance(value, dict):
        return {
            "macro_impact": 0,
            "stock_idea_density": 0,
            "contrarian_value": 0,
            "ai_infrastructure_relevance": 0,
            "show_utility": 0,
        }

    scores: dict[str, int] = {}
    for key in (
        "macro_impact",
        "stock_idea_density",
        "contrarian_value",
        "ai_infrastructure_relevance",
        "show_utility",
    ):
        raw = coerce_int(value.get(key))
        scores[key] = max(0, min(5, raw)) if raw is not None else 0

    return scores


def normalize_catalysts(value: Any) -> list[str]:
    """Normalize the catalysts array (list of plain strings)."""
    if not isinstance(value, list):
        return []
    catalysts: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            catalysts.append(text)
    return catalysts


def normalize_signal_extraction(value: Any) -> dict[str, Any]:
    """Normalize the full signal_extraction block."""
    if not isinstance(value, dict):
        value = {}

    return {
        "quick_verdict": str(value.get("quick_verdict") or "No signal analysis available.").strip(),
        "speaker_assessment": str(
            value.get("speaker_assessment") or "No speaker assessment available."
        ).strip(),
        "market_bias": normalize_market_bias(value.get("market_bias")),
        "macro_signals": normalize_macro_signals(value.get("macro_signals")),
        "tickers": normalize_tickers(value.get("tickers")),
        "catalysts": normalize_catalysts(value.get("catalysts")),
        "show_relevance": normalize_show_relevance(value.get("show_relevance")),
        "scores": normalize_scores(value.get("scores")),
    }


def normalize_summary(payload: dict[str, Any]) -> dict[str, Any]:
    takeaways = normalize_takeaways(payload.get("takeaways"))
    chapters = normalize_chapters(payload.get("chapters"))
    shorts = normalize_shorts(payload.get("shorts"))
    slide_suggestions = normalize_slide_suggestions(
        payload.get("slide_suggestions", payload.get("slides"))
    )

    if chapters and len(chapters) < 8:
        LOGGER.warning("Model returned %d chapters (expected 8-14)", len(chapters))
    if shorts and len(shorts) < 10:
        LOGGER.warning("Model returned %d shorts (expected 10-14)", len(shorts))

    return {
        "signal_extraction": normalize_signal_extraction(payload.get("signal_extraction")),
        "takeaways": takeaways,
        "chapters": chapters,
        "shorts": shorts,
        "slide_suggestions": slide_suggestions,
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        transcript_path = resolve_path(args.transcript)
        output_path = resolve_path(args.output)

        transcript_payload = load_json(transcript_path, default={})
        segments = normalize_transcript_segments(transcript_payload)
        if not segments:
            raise ContentScoutError(f"Transcript has no usable segments: {transcript_path}")

        annotations_payload: Any = []
        if args.annotations:
            annotations_path = resolve_path(args.annotations)
            annotations_payload = load_json(annotations_path, default=[])
        video_id = str(
            transcript_payload.get("videoId") or transcript_path.stem.replace("_transcript", "")
        ).strip()
        visual_annotations = normalize_annotations(annotations_payload, video_id=video_id)

        metadata = transcript_metadata(transcript_payload, transcript_path, segments)
        prompt = build_prompt(metadata, segments, visual_annotations)

        provider = detect_provider(args.provider)
        model = args.model or DEFAULT_MODELS[provider]
        LOGGER.info("Using provider=%s model=%s", provider, model)

        call_fn = call_anthropic if provider == "anthropic" else call_openai
        raw_response = call_fn(ANALYST_PERSONA, prompt, model)
        parsed = parse_json_object(raw_response)
        summary = normalize_summary(parsed)

        save_json(output_path, summary)
        LOGGER.info(
            "Summary written to %s (takeaways=%s chapters=%s shorts=%s slides=%s)",
            output_path,
            len(summary["takeaways"]),
            len(summary["chapters"]),
            len(summary["shorts"]),
            len(summary["slide_suggestions"]),
        )
        print(
            "Generated summary at "
            f"{output_path} via {provider}/{model} "
            f"(takeaways={len(summary['takeaways'])}, chapters={len(summary['chapters'])}, "
            f"shorts={len(summary['shorts'])})"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("summarize_video failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
