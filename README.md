# Transcript Studio + Content Scout Skills

YouTube video processing pipeline for OpenClaw agents.
- **content-scout**: Frame extraction, classification, daily briefs
- **transcript-studio**: Local transcription (mlx-whisper), speaker diarization, Notion export

## Install
```bash
git clone https://github.com/OpenCnid/transcript-studio.git ~/.openclaw/workspace/skills
pip3 install mlx-whisper pyannote-audio notion-client imagehash Pillow python-slugify anthropic openai yt-dlp
brew install ffmpeg
```
