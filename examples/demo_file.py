#!/usr/bin/env python3
"""音声ファイルを giziroku API に投げて結果を取得するデモ。

使い方:
    python demo_file.py sample.wav
    python demo_file.py meeting.m4a --diarize
    python demo_file.py sample.wav --language auto
    python demo_file.py meeting.wav --diarize --min-speakers 2 --max-speakers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


def submit(api: str, key: str, path: Path, diarize: bool, **form) -> str:
    url = f"{api}/{'transcribe-diarize' if diarize else 'transcribe'}"
    with path.open("rb") as f:
        files = {"file": (path.name, f)}
        data = {k: str(v) for k, v in form.items() if v is not None}
        r = requests.post(url, headers={"X-API-Key": key}, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()["job_id"]


def poll(api: str, key: str, job_id: str, interval: float = 2.0) -> dict:
    url = f"{api}/jobs/{job_id}"
    while True:
        r = requests.get(url, headers={"X-API-Key": key}, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data["status"]
        if status == "queued":
            print(f"  queued (position={data.get('queue_position', '?')})", file=sys.stderr)
        elif status == "running":
            print("  running...", file=sys.stderr)
        elif status == "succeeded":
            return data["result"]
        elif status == "failed":
            raise RuntimeError(f"job failed: {data.get('error')}")
        time.sleep(interval)


def render_transcribe(result: dict) -> None:
    print(f"[language={result['language']} duration={result['duration']:.1f}s]")
    print()
    for seg in result["segments"]:
        print(f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['text'].strip()}")
    print()
    print("==== full text ====")
    print(result["text"].strip())


def render_diarize(result: dict) -> None:
    print(f"[language={result['language']} duration={result['duration']:.1f}s]")
    print()
    for seg in result["segments"]:
        print(f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['speaker']:>12}: {seg['text'].strip()}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--api", default="http://localhost:8000", help="giziroku API base URL")
    p.add_argument("--key", default="change-me-please", help="X-API-Key value")
    p.add_argument("--language", default=None, help="ja / en / ... / auto")
    p.add_argument("--initial-prompt", default=None, help="固有名詞の補正ヒント (短く)")
    p.add_argument("--diarize", action="store_true", help="話者分離あり")
    p.add_argument("--num-speakers", type=int, default=None)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    args = p.parse_args()

    if not args.audio.is_file():
        print(f"file not found: {args.audio}", file=sys.stderr)
        return 1

    print(f"uploading {args.audio} ({args.audio.stat().st_size / 1024:.0f} KB)...", file=sys.stderr)
    form = {"language": args.language, "initial_prompt": args.initial_prompt}
    if args.diarize:
        form.update(
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
    job_id = submit(args.api, args.key, args.audio, args.diarize, **form)
    print(f"job_id={job_id}", file=sys.stderr)

    result = poll(args.api, args.key, job_id)
    print(file=sys.stderr)
    (render_diarize if args.diarize else render_transcribe)(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
