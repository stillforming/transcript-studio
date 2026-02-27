---
name: content-scout
description: >
  YouTube channel monitoring and daily content briefing pipeline. Monitors configured
  channels for new uploads, downloads videos, extracts/classifies visual frames (charts,
  slides, screens vs talking heads), transcribes audio, and generates a daily markdown
  brief with key takeaways. Use when: (1) processing YouTube videos for visual and
  transcript analysis, (2) generating daily content briefs from monitored channels,
  (3) running the content-scout pipeline or any of its steps, (4) managing channel
  watchlists, (5) frame extraction or classification tasks. Requires: yt-dlp, ffmpeg,
  Python 3.10+, PIL/Pillow, imagehash, python-slugify. Optional: OpenAI API (transcription
  fallback), notion-client (Notion sync).
---

# Content Scout

YouTube monitoring → frame extraction → classification → transcription → daily brief.

## ⚠️ Success Criteria — VERIFY BEFORE REPORTING DONE

**The pipeline is complete ONLY when the final JSON output contains `"verdict": "COMPLETE"`.**

After the pipeline finishes, verify:

```bash
python3 -c "
import json
log = json.load(open('content-vault/processing-log.json'))
print(f'Processed: {len(log[\"processedVideoIds\"])} videos')
print(f'Recent IDs: {log[\"processedVideoIds\"][-3:]}')
"
```

**Required checks:**
1. Pipeline JSON output shows `"verdict": "COMPLETE"`
2. `content-vault/processing-log.json` includes the video ID in `processedVideoIds`
3. `content-vault/daily/YYYY-MM-DD/` contains the daily brief markdown

**If any check fails, do NOT report success.** Report which steps failed and why.

## Pipeline Steps

Run the full pipeline or individual steps. The orchestrator is `scripts/run_pipeline.py`.

```bash
# Full daily pipeline
python3 scripts/run_pipeline.py --date 2026-02-24

# Single video
python3 scripts/run_pipeline.py --video-url "https://youtube.com/watch?v=VIDEO_ID"

# Resume from a specific step
python3 scripts/run_pipeline.py --date 2026-02-24 --resume-from classify
```

### Step Order

1. **select** — `select_videos.py` — pick videos from monitored channels (config/channels.json)
2. **download** — `download.py` — download via yt-dlp (720p max, delays between downloads)
3. **extract** — `extract_frames.py` — extract frames at configured interval, deduplicate via perceptual hash
4. **transcribe** — `transcribe.py` — transcribe audio via OpenAI Whisper API
5. **window** — `window_transcripts.py` — split transcript into 30s windows
6. **classify** — `classify_annotate.py` — LLM classifies each frame (CHART_VISUAL, SLIDE, SCREEN, GRAPHIC, TALKING_HEAD)
7. **compress** — `compress_and_store.py` — compress kept frames to webp, store in content-vault
8. **tags** — `build_tag_index.py` — build searchable tag index from metadata
9. **log** — `update_log.py` — update processing log
10. **notion** — `notion_sync.py` — sync to Notion databases (optional, needs NOTION_TOKEN)
11. **brief** — `generate_brief.py` — generate daily markdown brief with takeaways
12. **archive** / **cleanup** — archive transcripts, clean tmp files

## Configuration

All config lives in `config/`:

- `channels.json` — monitored YouTube channels with priority scores
- `keywords.json` — keyword relevance scoring
- `watchlist.json` — additional video URLs to process
- `settings.json` — pipeline parameters (frame interval, thresholds, limits, cron schedule)

## Working Directory

Set `CONTENT_SCOUT_WORKDIR` to control where output goes. Defaults to `cwd`.
Config files (`config/`) always resolve from the skill installation directory.

```bash
export CONTENT_SCOUT_WORKDIR=/path/to/project
python3 scripts/run_pipeline.py --date 2026-02-24
```

Output structure (created in workdir, NOT in the skill directory):
```
content-vault/           # Output directory (created automatically)
├── daily/YYYY-MM-DD/    # Frames and briefs per day
├── channels/            # Per-channel metadata
├── tags/                # Tag index
└── processing-log.json  # Run history
tmp/                     # Working directory (cleaned up after pipeline)
├── audio/               # Extracted audio files
├── frames/              # Raw extracted frames
└── transcripts/         # Whisper output
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | For transcription | Whisper API transcription |
| `NOTION_TOKEN` | For Notion sync | Notion API integration |
| `ANTHROPIC_API_KEY` | For classification | Frame classification via Claude |

## Key Utilities (_common.py)

Shared utilities used by all scripts: `resolve_path`, `load_json`, `save_json`, `ensure_dir`,
`setup_logging`, `run_command`, `ROOT_DIR`, `ContentScoutError`, `normalize_slug`, `utcnow`,
`utc_today_str`, `parse_upload_date`.

`ROOT_DIR` resolves to the skill's parent directory at runtime (two levels up from scripts/).
All path resolution is relative to ROOT_DIR.

## Dependencies

```
yt-dlp
ffmpeg (system)
Pillow
imagehash
python-slugify
openai (optional — transcription)
anthropic (optional — classification)
notion-client (optional — Notion sync)
```
