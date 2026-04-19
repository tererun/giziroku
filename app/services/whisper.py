from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from faster_whisper import WhisperModel

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    segments: List[Segment]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments).strip()


class WhisperService:
    def __init__(self) -> None:
        s = get_settings()
        log.info("Loading faster-whisper model=%s device=%s compute=%s",
                 s.whisper_model, s.device, s.whisper_compute_type)
        self._model = WhisperModel(
            s.whisper_model,
            device=s.device,
            compute_type=s.whisper_compute_type,
        )
        self._default_language = s.default_language
        self._default_initial_prompt = s.default_initial_prompt or None

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        lang = language or self._default_language
        lang_arg = None if lang == "auto" else lang
        prompt = initial_prompt if initial_prompt is not None else self._default_initial_prompt

        segments_iter, info = self._model.transcribe(
            audio,
            language=lang_arg,
            beam_size=5,
            vad_filter=vad_filter,
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        segments = [Segment(start=s.start, end=s.end, text=s.text) for s in segments_iter]
        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )

    def transcribe_segment(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> str:
        """Short helper returning just text — used for per-speaker clips."""
        res = self.transcribe(audio, language=language, vad_filter=False)
        return res.text
