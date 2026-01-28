# api/main.py
from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

from api import jobs
from api.jobs import JobStatus
from api.schemas import (
    AttemptRequest,
    AttemptResponse,
    AsrCreateResponse,
    AsrJobResponse,
    HealthResponse,
    QuizRequest,
    QuizResponse,
    YtTranscriptRequest,
    YtTranscriptResponse,
    YtQuizRequest,
    RepeatSimilarRequest,
)
from api.youtube_transcripts import extract_video_id, fetch_youtube_transcript
from api.adapters import run_asr, run_grading, run_quiz

app = FastAPI(title="Listening Coach API", version="0.1.0")

MODEL_ID = "Sparkplugx1904/whisper-base-id"
ASR_PIPE: Any | None = None

def build_asr_pipe() -> Any:
    # Streamlit 캐시 대신 FastAPI startup에서 1회 로드
    return pipeline("automatic-speech-recognition", model=MODEL_ID)

@app.on_event("startup")
def _load_models():
    global ASR_PIPE
    ASR_PIPE = build_asr_pipe()

# 필요 시 Streamlit/Next.js 도메인으로 좁히세요(지금은 개발 편의상 all 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"ok": True}


def _asr_job(job_id: str, audio_bytes: bytes, filename: str) -> None:
    try:
        jobs.set_status(job_id, JobStatus.running)

        if ASR_PIPE is None:
            raise RuntimeError("ASR_PIPE is not initialized. Load ASR model/pipe at startup.")

        transcript = run_asr(asr_pipe=ASR_PIPE, audio_bytes=audio_bytes, filename=filename)
        jobs.set_result(job_id, {"transcript": transcript})

    except Exception as e:
        jobs.set_error(job_id, str(e))


@app.post("/asr", response_model=AsrCreateResponse)
async def create_asr_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # UploadFile은 form-data 기반이며 python-multipart 설치가 필요합니다. :contentReference[oaicite:4]{index=4}
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    job_id = jobs.create_job()
    background_tasks.add_task(_asr_job, job_id, audio_bytes, file.filename or "audio")
    return {"job_id": job_id}


@app.get("/asr/{job_id}", response_model=AsrJobResponse)
def get_asr_job(job_id: str):
    try:
        rec = jobs.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")

    payload = {"job_id": rec.job_id, "status": rec.status.value}
    if rec.status == JobStatus.done and rec.result:
        payload["transcript"] = rec.result.get("transcript")
    if rec.status == JobStatus.failed:
        payload["error"] = rec.error
    return payload


@app.post("/quiz", response_model=QuizResponse)
def create_quiz(req: QuizRequest):
    try:
        quiz = run_quiz(transcript=req.transcript, level=req.level)
        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/yt/transcript", response_model=YtTranscriptResponse)
def yt_transcript(req: YtTranscriptRequest):
    try:
        vid = extract_video_id(req.url)  # ✅ 실패 시 여기서 예외 발생
        transcript = fetch_youtube_transcript(
            req.url,
            prefer_langs=((req.lang or "id"), "en"),
        )
        return {"video_id": vid, "transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/yt/quiz", response_model=QuizResponse)
def yt_quiz(req: YtQuizRequest):
    """
    YouTube URL -> 자막 자동 추출 -> quiz 생성
    """
    try:
        vid = extract_video_id(req.url)
        transcript = fetch_youtube_transcript(
            vid,
            preferred_lang=(req.lang or "id"),
        )
        quiz = run_quiz(transcript=transcript, level=req.level)
        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/repeat/similar", response_model=QuizResponse)
def repeat_similar(req: RepeatSimilarRequest):
    try:
        prev = req.prev_quiz or {}
        qs = prev.get("questions", []) if isinstance(prev, dict) else []

        # 이전 퀴즈를 “피해야 할 목록”으로 만들기 (프롬프트 길이 폭주 방지로 컷)
        avoid_lines = []
        for q in qs:
            qid = q.get("id")
            qtext = str(q.get("question", "")).strip()
            ev = str(q.get("evidence_quote", "")).strip()
            if qtext:
                avoid_lines.append(f"- Q{qid}: {qtext}")
            if ev:
                avoid_lines.append(f"  evidence: {ev}")

        avoid_blob = "\n".join(avoid_lines)
        if len(avoid_blob) > 1200:
            avoid_blob = avoid_blob[:1200] + "\n...(truncated)"

        extra_constraint = (
            "REPEAT_SIMILAR_CONSTRAINT:\n"
            "Create a NEW quiz that tests the SAME transcript/skills but is DIFFERENT from the previous quiz.\n"
            "Do NOT reuse the same questions, do NOT reuse the same evidence quotes, and do NOT copy previous distractors.\n"
            "Avoid these previous items:\n"
            f"{avoid_blob}\n"
        )

        quiz = run_quiz(
            transcript=req.transcript,
            level=req.level,
            extra_constraint=extra_constraint,
        )
        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attempts", response_model=AttemptResponse)
def create_attempt(req: AttemptRequest):
    try:
        graded = run_grading(quiz=req.quiz, user_answers=req.user_answers)
        # graded 반환 형태가 프로젝트마다 다르므로 최대한 유연하게 담습니다.
        score = graded.get("score")
        wrong_items = graded.get("wrong_items", [])
        feedback = graded.get("feedback")
        return {"score": score, "wrong_items": wrong_items, "feedback": feedback, "raw": graded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
