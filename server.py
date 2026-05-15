"""
Resident Whisper HTTP server.
Loads the model once at startup, accepts audio via POST /transcribe.
"""
import os
import tempfile
from contextlib import asynccontextmanager

import whisper
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

MODEL_NAME = os.environ.get("WHISPER_MODEL", "turbo")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
INITIAL_PROMPT = os.environ.get(
    "WHISPER_INITIAL_PROMPT",
    "Разработчик диктует команды и заметки. Используй правильную пунктуацию. Английские термины пиши латиницей: git, deploy, branch, merge, commit, docker, kubernetes, API, SQL.",
)

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE}...", flush=True)
    _model = whisper.load_model(MODEL_NAME, device=DEVICE)
    print("Model ready.", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
):
    # Empty string or "auto" → let Whisper detect language automatically.
    # Useful for mixed Russian/English speech.
    lang = language.strip() or None
    if lang == "auto":
        lang = None

    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = _model.transcribe(tmp_path, language=lang, initial_prompt=INITIAL_PROMPT)
        text = result["text"].strip()
    finally:
        os.unlink(tmp_path)

    return JSONResponse({"text": text})


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}
