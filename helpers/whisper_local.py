"""Local, free transcription with whisper.cpp, in the shape Scribe returns.

Scribe is the only paid step of video-use. This backend runs whisper.cpp on the
machine instead and converts its word-level output into the same JSON the rest
of the helpers already read (`words[]` with `text`, `start`, `end`, `type`,
`speaker_id`), so pack_transcripts, timeline_view and render need no change.

What you lose compared to Scribe:
  - no speaker diarization: every word is `speaker_0`;
  - no audio events (`(laughter)`, `(applause)`);
  - fillers (`euh`, `umm`) are often smoothed away. The default prompt asks for
    them, which helps, but a cut list built on fillers should be checked.

Setup:
    brew install whisper-cpp
    curl -L -o ~/.cache/whisper/ggml-small.bin \\
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin

Environment:
    WHISPER_MODEL   path to a ggml model (default ~/.cache/whisper/ggml-small.bin)
    WHISPER_BIN     whisper.cpp CLI (default: whisper-cli on PATH)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_MODEL = Path.home() / ".cache" / "whisper" / "ggml-small.bin"

# A prompt written the way people talk nudges whisper to keep the fillers the
# editor needs to cut. It is a style hint, not text to transcribe.
FILLER_PROMPT = {
    "fr": "Euh, bon, alors, hum... Voilà, en fait, tu vois.",
    "en": "Umm, uh, so, like... you know, I mean.",
}


def whisper_available() -> bool:
    return bool(shutil.which(os.environ.get("WHISPER_BIN", "whisper-cli"))) and model_path().exists()


def model_path() -> Path:
    return Path(os.environ.get("WHISPER_MODEL", DEFAULT_MODEL)).expanduser()


def to_scribe(whisper_json: dict) -> dict:
    """Convert whisper.cpp `-ml 1 -sow -oj` output to Scribe's response shape.

    With max-len 1 and split-on-word, each transcription segment is one word
    with its millisecond offsets. Empty segments (leading silence) are dropped;
    a `spacing` entry sits between consecutive words, as in Scribe.
    """
    words: list[dict] = []
    for seg in whisper_json.get("transcription", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg["offsets"]["from"] / 1000.0
        end = seg["offsets"]["to"] / 1000.0
        if words:
            prev_end = words[-1]["end"]
            words.append({"text": " ", "start": prev_end, "end": max(prev_end, start),
                          "type": "spacing", "speaker_id": "speaker_0"})
        words.append({"text": text, "start": start, "end": end,
                      "type": "word", "speaker_id": "speaker_0"})
    language = (whisper_json.get("result") or {}).get("language") or ""
    return {
        "language_code": language,
        "text": " ".join(w["text"] for w in words if w["type"] == "word"),
        "words": words,
        "transcriber": "whisper.cpp",
    }


def call_whisper(audio_path: Path, language: str | None = None, prompt: str | None = None) -> dict:
    """Transcribe a 16 kHz mono WAV with whisper.cpp; returns a Scribe-shaped dict.

    `prompt` adds vocabulary (names, brands) on top of the filler hint, e.g.
    "Awa, Baarali, PersonnIA": whisper otherwise spells an unknown name the way
    it sounds.
    """
    model = model_path()
    if not model.exists():
        raise RuntimeError(f"whisper model not found: {model} (see helpers/whisper_local.py)")
    hint = FILLER_PROMPT.get(language or "", "")
    full_prompt = " ".join(p for p in (prompt, hint) if p)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "out"
        cmd = [
            os.environ.get("WHISPER_BIN", "whisper-cli"),
            "-m", str(model),
            "-f", str(audio_path),
            "-l", language or "auto",
            "-ml", "1", "-sow",      # one segment per word
            "-oj", "-of", str(base),
            "-np",
        ]
        if full_prompt:
            cmd += ["--prompt", full_prompt]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"whisper.cpp failed: {res.stderr[-500:]}")
        data = json.loads(Path(f"{base}.json").read_text())
    return to_scribe(data)
