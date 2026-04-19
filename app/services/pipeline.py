from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.services.diarize import DiarizationSegment, DiarizationService
from app.services.whisper import Segment, TranscriptionResult, WhisperService


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class DiarizedResult:
    language: str
    duration: float
    segments: List[SpeakerSegment] = field(default_factory=list)


def _majority_speaker(
    start: float, end: float, diar: List[DiarizationSegment]
) -> str:
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for d in diar:
        if d.end <= start:
            continue
        if d.start >= end:
            break
        overlap = min(end, d.end) - max(start, d.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = d.speaker
    return best_speaker


def merge_transcription_with_diarization(
    transcription: TranscriptionResult,
    diarization: List[DiarizationSegment],
) -> DiarizedResult:
    """Assign a speaker label to each whisper segment by max overlap."""
    out: List[SpeakerSegment] = []
    for seg in transcription.segments:
        speaker = _majority_speaker(seg.start, seg.end, diarization)
        if out and out[-1].speaker == speaker:
            # merge with previous for readability
            out[-1] = SpeakerSegment(
                start=out[-1].start,
                end=seg.end,
                speaker=speaker,
                text=(out[-1].text + seg.text),
            )
        else:
            out.append(SpeakerSegment(
                start=seg.start, end=seg.end, speaker=speaker, text=seg.text,
            ))
    return DiarizedResult(
        language=transcription.language,
        duration=transcription.duration,
        segments=out,
    )


class TranscribeDiarizePipeline:
    def __init__(self, whisper: WhisperService, diarizer: DiarizationService) -> None:
        self._whisper = whisper
        self._diarizer = diarizer

    def run(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizedResult:
        transcription = self._whisper.transcribe(audio, language=language)
        diarization = self._diarizer.diarize(
            audio,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return merge_transcription_with_diarization(transcription, diarization)
