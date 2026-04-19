from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.auth import require_api_key
from app.schemas import (
    DiarizedResult,
    DiarizedSegment,
    JobResponse,
    JobStatusResponse,
    TranscribeResult,
    TranscribeSegment,
)
from app.utils.audio import decode_to_pcm16

log = logging.getLogger(__name__)
router = APIRouter(tags=["transcribe"], dependencies=[Depends(require_api_key)])


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload").suffix or ".bin"
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while chunk := await file.read(1024 * 1024):
            fd.write(chunk)
    finally:
        fd.close()
    return Path(fd.name)


def _transcribe_to_schema(res) -> TranscribeResult:
    return TranscribeResult(
        language=res.language,
        language_probability=res.language_probability,
        duration=res.duration,
        text=res.text,
        segments=[TranscribeSegment(start=s.start, end=s.end, text=s.text) for s in res.segments],
    )


def _diarized_to_schema(res) -> DiarizedResult:
    return DiarizedResult(
        language=res.language,
        duration=res.duration,
        segments=[
            DiarizedSegment(start=s.start, end=s.end, speaker=s.speaker, text=s.text)
            for s in res.segments
        ],
    )


@router.post("/transcribe", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    initial_prompt: Optional[str] = Form(default=None),
):
    """話者分離なしの文字起こし。ジョブをキューに投入して job_id を返す。"""
    path = await _save_upload(file)
    whisper = request.app.state.whisper
    queue = request.app.state.queue

    async def _task():
        try:
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(None, decode_to_pcm16, path)
            res = await loop.run_in_executor(
                None, lambda: whisper.transcribe(audio, language=language, initial_prompt=initial_prompt),
            )
            return _transcribe_to_schema(res).model_dump()
        finally:
            path.unlink(missing_ok=True)

    try:
        job = await queue.submit(_task)
    except asyncio.QueueFull:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="queue is full")
    return JobResponse(job_id=job.id, status=job.status, queue_position=job.queue_position)


@router.post("/transcribe-diarize", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_transcribe_diarize(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    initial_prompt: Optional[str] = Form(default=None),
    num_speakers: Optional[int] = Form(default=None),
    min_speakers: Optional[int] = Form(default=None),
    max_speakers: Optional[int] = Form(default=None),
):
    """話者分離あり。pyannote で分離 → whisper で各区間を含めて文字起こし。"""
    path = await _save_upload(file)
    pipeline = request.app.state.pipeline
    queue = request.app.state.queue

    async def _task():
        try:
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(None, decode_to_pcm16, path)
            res = await loop.run_in_executor(
                None,
                lambda: pipeline.run(
                    audio,
                    language=language,
                    initial_prompt=initial_prompt,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                ),
            )
            return _diarized_to_schema(res).model_dump()
        finally:
            path.unlink(missing_ok=True)

    try:
        job = await queue.submit(_task)
    except asyncio.QueueFull:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="queue is full")
    return JobResponse(job_id=job.id, status=job.status, queue_position=job.queue_position)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, request: Request):
    job = request.app.state.queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found or expired")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        queue_position=job.queue_position,
        error=job.error,
        result=job.result if job.status == "succeeded" else None,
    )
