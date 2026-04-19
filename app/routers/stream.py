from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.auth import require_api_key_ws
from app.config import get_settings
from app.services.pipeline import merge_transcription_with_diarization
from app.utils.audio import pcm16_bytes_to_float32

log = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])


async def _receive_pcm_chunk(
    ws: WebSocket, target_samples: int, sample_rate: int
) -> Optional[np.ndarray]:
    """Accumulate 16-bit mono PCM from the client until we have `target_samples`.

    Client sends binary frames (raw PCM s16le @ sample_rate) or a text "flush"
    to force-emit whatever is buffered.
    """
    buffer = bytearray()
    while True:
        try:
            msg = await ws.receive()
        except WebSocketDisconnect:
            return None
        if msg["type"] == "websocket.disconnect":
            return None
        if (data := msg.get("bytes")) is not None:
            buffer.extend(data)
            if len(buffer) >= target_samples * 2:
                return pcm16_bytes_to_float32(bytes(buffer))
        elif (text := msg.get("text")) is not None:
            if text.strip().lower() in ("flush", "eof"):
                if not buffer:
                    return np.zeros(0, dtype=np.float32)
                return pcm16_bytes_to_float32(bytes(buffer))
            # ignore other text commands


@router.websocket("/stream/transcribe")
async def ws_transcribe(
    websocket: WebSocket,
    language: Optional[str] = Query(default=None),
    api_key: Optional[str] = Query(default=None),
):
    """Realtime transcription.

    Protocol:
      - client opens WS with ?api_key=xxx (or X-API-Key header) and optional ?language=ja
      - client streams raw 16-bit mono PCM at STREAM_SAMPLE_RATE (default 16k)
      - server emits JSON `{type: "partial", text, segments, elapsed}` per chunk
      - client sends text "flush" to end; server replies `{type: "final"}`
    """
    if not await require_api_key_ws(websocket, api_key):
        return
    await websocket.accept()

    s = get_settings()
    whisper = websocket.app.state.whisper
    queue = websocket.app.state.queue
    chunk_samples = int(s.stream_chunk_seconds * s.stream_sample_rate)
    elapsed = 0.0

    try:
        while True:
            audio = await _receive_pcm_chunk(websocket, chunk_samples, s.stream_sample_rate)
            if audio is None:
                return
            if audio.size == 0:
                await websocket.send_json({"type": "final"})
                await websocket.close()
                return

            loop = asyncio.get_running_loop()
            async with queue.gpu_lock:
                res = await loop.run_in_executor(
                    None, whisper.transcribe, audio, language
                )

            await websocket.send_json({
                "type": "partial",
                "text": res.text,
                "language": res.language,
                "segments": [
                    {"start": elapsed + s_.start, "end": elapsed + s_.end, "text": s_.text}
                    for s_ in res.segments
                ],
                "offset": elapsed,
            })
            elapsed += audio.size / s.stream_sample_rate
    except WebSocketDisconnect:
        return


@router.websocket("/stream/transcribe-diarize")
async def ws_transcribe_diarize(
    websocket: WebSocket,
    language: Optional[str] = Query(default=None),
    api_key: Optional[str] = Query(default=None),
):
    """Realtime transcription + speaker diarization per buffered chunk.

    pyannote は本来オフライン用途のため、チャンクごとに話者ラベルを振り直します。
    話者IDはチャンクをまたいで一致しません (SPEAKER_00 が毎回同じ人とは限らない)。
    """
    if not await require_api_key_ws(websocket, api_key):
        return
    await websocket.accept()

    s = get_settings()
    whisper = websocket.app.state.whisper
    diarizer = websocket.app.state.diarizer
    queue = websocket.app.state.queue
    chunk_samples = int(s.stream_chunk_seconds * s.stream_sample_rate)
    elapsed = 0.0

    try:
        while True:
            audio = await _receive_pcm_chunk(websocket, chunk_samples, s.stream_sample_rate)
            if audio is None:
                return
            if audio.size == 0:
                await websocket.send_json({"type": "final"})
                await websocket.close()
                return

            loop = asyncio.get_running_loop()
            async with queue.gpu_lock:
                transcription = await loop.run_in_executor(
                    None, whisper.transcribe, audio, language
                )
                diar = await loop.run_in_executor(None, diarizer.diarize, audio)
            merged = merge_transcription_with_diarization(transcription, diar)

            await websocket.send_json({
                "type": "partial",
                "language": merged.language,
                "offset": elapsed,
                "segments": [
                    {
                        "start": elapsed + seg.start,
                        "end": elapsed + seg.end,
                        "speaker": seg.speaker,
                        "text": seg.text,
                    }
                    for seg in merged.segments
                ],
            })
            elapsed += audio.size / s.stream_sample_rate
    except WebSocketDisconnect:
        return
