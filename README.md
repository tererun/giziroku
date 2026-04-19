# giziroku

Whisper + pyannote を載せた、文字起こし / 話者分離 API。GTX 1070 8GB クラスの GPU で動くよう `faster-whisper` + `int8_float16` で軽量に組んでいます。

- 話者分離なし / あり の両対応
- ファイルアップロード (非同期ジョブ) / WebSocket リアルタイム の両対応
- API キー認証 (`X-API-Key` もしくは WS なら `?api_key=`)
- シンプルな単一ワーカーキュー (GPU を直列化)

## 必要なもの

- Ubuntu + NVIDIA ドライバ (GTX 1070 以降)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- HuggingFace アカウントとアクセストークン
  - https://huggingface.co/pyannote/speaker-diarization-3.1 と https://huggingface.co/pyannote/segmentation-3.0 で "Agree and access repository" を押す
  - https://huggingface.co/settings/tokens でトークン発行 (無料)

## セットアップ

```bash
cp .env.example .env
# .env を編集: HF_TOKEN, API_KEYS を設定
docker compose build
docker compose up -d
docker compose logs -f giziroku   # 初回はモデルDLで数分かかります
```

モデルは `./models/` にキャッシュされます (次回以降は即起動)。

## エンドポイント

### ヘルスチェック

```
GET /health
```

### ファイルアップロード (非同期)

ジョブを投入 → `job_id` でポーリング。

**話者分離なし:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "X-API-Key: change-me-please" \
  -F "file=@sample.wav" \
  -F "language=ja" \
  -F "initial_prompt=giziroku, pyannote, faster-whisper, Docker"
```

- `language` 省略 or `auto` で自動判定
- `initial_prompt` は固有名詞の補正ヒント。Whisperが最後の~224トークンしか見ないので短く (長く渡しても後ろから切られるだけ)。常用したい語彙は `.env` の `DEFAULT_INITIAL_PROMPT` に入れておけば毎回付与される

**話者分離あり:**
```bash
curl -X POST http://localhost:8000/transcribe-diarize \
  -H "X-API-Key: change-me-please" \
  -F "file=@meeting.m4a" \
  -F "language=ja" \
  -F "min_speakers=2" -F "max_speakers=4"
```

**結果取得:**
```bash
curl -H "X-API-Key: change-me-please" http://localhost:8000/jobs/<job_id>
```

ステータスは `queued` → `running` → `succeeded`/`failed`。`succeeded` 時に `result` が返ります。

### WebSocket リアルタイム

16bit モノラル PCM (デフォルト 16kHz) のバイナリフレームを流し込みます。サーバは約 `STREAM_CHUNK_SECONDS` (既定 5秒) ごとに結果を JSON で返します。終了時はテキスト `flush` を送信。

- `ws://localhost:8000/stream/transcribe?api_key=...&language=ja&initial_prompt=...`
- `ws://localhost:8000/stream/transcribe-diarize?api_key=...&language=ja&initial_prompt=...`

無音判定 (`STREAM_SILENCE_RMS` 未満) のチャンクは whisper を呼ばず `{"type":"silence"}` を返します。GPU 使用量とレスポンス遅延を節約できます。

**Python クライアント例:**
```python
import asyncio, json, websockets, soundfile as sf, numpy as np

async def main():
    uri = "ws://localhost:8000/stream/transcribe?api_key=change-me-please&language=ja"
    audio, sr = sf.read("sample.wav", dtype="int16")
    assert sr == 16000 and audio.ndim == 1
    async with websockets.connect(uri) as ws:
        for i in range(0, len(audio), 16000):   # 1秒ずつ送信
            await ws.send(audio[i:i+16000].tobytes())
        await ws.send("flush")
        async for msg in ws:
            print(json.loads(msg))

asyncio.run(main())
```

**注意:** リアルタイム話者分離はチャンク単位で pyannote を回すため、`SPEAKER_00` がチャンクをまたいで同じ人とは限りません。正確な話者ラベルが必要ならファイルアップロードの `/transcribe-diarize` を使ってください。

## チューニング

`.env` の主な項目:

| 変数 | 用途 |
|---|---|
| `WHISPER_MODEL` | `tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3` |
| `WHISPER_COMPUTE_TYPE` | Pascal (GTX10xx): `int8_float32` 推奨 / `int8` / `float32`。Volta以降: `int8_float16` 推奨 |
| `DEFAULT_LANGUAGE` | `ja` (既定) / `en` / `auto` |
| `MAX_QUEUE_SIZE` | 溢れたら 503 |
| `STREAM_CHUNK_SECONDS` | 小さいほど低遅延だが精度が落ちる |
| `STREAM_SILENCE_RMS` | 無音判定閾値。大きくすると小声もスキップ、小さくすると雑音で誤反応 |
| `DEFAULT_INITIAL_PROMPT` | 毎回 Whisper に渡す固有名詞ヒント |

GTX 1070 (8GB) での目安:
- `large-v3` + pyannote 同時ロード → 約 4〜5GB 使用、余裕あり
- VRAM 不足で OOM したら `medium` か `WHISPER_COMPUTE_TYPE=int8` に落とす

## アーキテクチャ

```
FastAPI
 ├── /transcribe, /transcribe-diarize (upload)   ─┐
 ├── /jobs/{id}                                   ├─ JobQueue (単一worker, GPU lock)
 └── /stream/... (WebSocket)                     ─┘   └─ WhisperService / DiarizationService
```

GPU ロック 1 本で文字起こし・分離・ストリーミングを直列化しています。1070 1枚で十分まわる一方、同時に裁ける並列度は実質 1。必要になったら Redis + worker 複数台への置き換えを検討してください。
