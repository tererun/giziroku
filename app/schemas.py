from typing import List, Optional

from pydantic import BaseModel, Field


class TranscribeSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResult(BaseModel):
    language: str
    language_probability: float
    duration: float
    text: str
    segments: List[TranscribeSegment]


class DiarizedSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class DiarizedResult(BaseModel):
    language: str
    duration: float
    segments: List[DiarizedSegment]


class JobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int = 0


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int = 0
    error: Optional[str] = None
    result: Optional[dict] = None


class TranscribeOptions(BaseModel):
    language: Optional[str] = Field(
        default=None,
        description="ja / en / ... / auto。省略時は設定のデフォルト。",
    )


class DiarizeOptions(TranscribeOptions):
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
