import subprocess
from pathlib import Path

import numpy as np


TARGET_SR = 16000


def decode_to_pcm16(src: Path, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 PCM at `sample_rate`."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(src),
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def pcm16_bytes_to_float32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
