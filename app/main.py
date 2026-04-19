from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.queue import JobQueue
from app.routers import stream, transcribe
from app.services.diarize import DiarizationService
from app.services.pipeline import TranscribeDiarizePipeline
from app.services.whisper import WhisperService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("giziroku")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Starting giziroku (device=%s model=%s)", settings.device, settings.whisper_model)

    app.state.whisper = WhisperService()
    app.state.diarizer = DiarizationService()
    app.state.pipeline = TranscribeDiarizePipeline(app.state.whisper, app.state.diarizer)
    app.state.queue = JobQueue()
    await app.state.queue.start()

    log.info("giziroku ready")
    try:
        yield
    finally:
        await app.state.queue.stop()


app = FastAPI(
    title="giziroku",
    description="Whisper + pyannote 文字起こし / 話者分離 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(transcribe.router)
app.include_router(stream.router)


@app.get("/health")
async def health():
    s = get_settings()
    return {
        "status": "ok",
        "model": s.whisper_model,
        "device": s.device,
        "language_default": s.default_language,
    }
