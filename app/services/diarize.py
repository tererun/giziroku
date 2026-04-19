from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from pyannote.audio import Pipeline

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


class DiarizationService:
    def __init__(self) -> None:
        s = get_settings()
        if not s.hf_token:
            raise RuntimeError("HF_TOKEN is required for pyannote diarization")
        log.info("Loading pyannote speaker-diarization-3.1")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=s.hf_token,
        )
        if s.device == "cuda":
            self._pipeline.to(torch.device("cuda"))
        self._sample_rate = 16000

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[DiarizationSegment]:
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        annotation = self._pipeline(
            {"waveform": waveform, "sample_rate": self._sample_rate},
            **kwargs,
        )
        segments: List[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            ))
        segments.sort(key=lambda s: s.start)
        return segments
