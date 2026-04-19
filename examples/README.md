# giziroku クライアントデモ

Mac (or 任意の PC) から giziroku API を叩くサンプル。

## セットアップ

```bash
cd examples
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sounddevice` は PortAudio に依存します。Mac は `brew install portaudio` しておいてください。

## サーバURL

デフォルトは `http://localhost:8000`。別ホスト (例: `tererunsrv03.local:8000`) の場合は `--api` で指定。

```bash
# REST
python demo_file.py sample.wav --api http://tererunsrv03.local:8000 --key <API_KEY>

# WebSocket
python demo_mic.py --api ws://tererunsrv03.local:8000 --key <API_KEY>
```

## ファイルデモ (demo_file.py)

音声ファイルをアップロードして結果を取得。

```bash
# 文字起こしのみ
python demo_file.py sample.wav

# 話者分離あり (話者数ヒント付き)
python demo_file.py meeting.m4a --diarize --min-speakers 2 --max-speakers 4

# 言語自動判定
python demo_file.py foreign.wav --language auto
```

対応形式: ffmpeg で読めれば wav / mp3 / m4a / flac / mp4 など何でも。

## マイクデモ (demo_mic.py)

Mac のマイクから 16kHz mono PCM を流し込んでリアルタイム文字起こし。

```bash
# 入力デバイス一覧確認
python demo_mic.py --list

# 文字起こし (ja)
python demo_mic.py

# 話者分離あり
python demo_mic.py --diarize

# 外部マイク指定
python demo_mic.py --device 2
```

`Ctrl+C` で停止 (サーバに `flush` を送信してクローズ)。

**既知の制約:**
- サーバ側で約 `STREAM_CHUNK_SECONDS` (既定5秒) バッファしてから文字起こしするので、画面に出るまで数秒遅延します
- リアルタイム話者分離はチャンクごとに再計算するため、`SPEAKER_00` が同じ人を指し続ける保証はありません
