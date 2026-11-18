"""Offline word timings for the content-cue view.

The detector reads only the audio, but one of its feature views needs word
timings ("this message comes from", "use code"). SPEC now allows a lightweight
offline ASR for exactly that: no network calls, no LLM, no external transcript.
This module is the only place that touches an ASR engine; everything after it
consumes plain segment dicts, so training (pipeline transcript) and deployment
(small local model) share one code path.

The engine is an optional dependency, like scikit-learn for training: the
detector stays importable without it and fails with a clear message only when a
speech-cue model is actually used.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_ASR_MODEL = "tiny"
DEFAULT_LANGUAGE = "en"


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe_words(
    src: str,
    model: str = DEFAULT_ASR_MODEL,
    language: str = DEFAULT_LANGUAGE,
    device: str = "cpu",
    compute_type: str = "int8",
    proxy: Optional[str] = None,
) -> list[dict]:
    """Return ``[{"start_s", "end_s", "text"}, ...]`` for one recording.

    ``model`` may be a size name (downloaded once to the local cache) or a path
    to a pre-downloaded model directory, which is what an offline deployment
    should pass.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "内容线索模型需要离线 ASR；请先 pip install faster-whisper，"
            "或用 --transcripts 直接提供词级转写"
        ) from error

    engine = WhisperModel(model, device=device, compute_type=compute_type)
    segments, _ = engine.transcribe(src, language=language, vad_filter=True)
    rows = []
    for segment in segments:
        text = str(getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "start_s": float(segment.start),
                "end_s": float(segment.end),
                "text": text,
            }
        )
    return rows
