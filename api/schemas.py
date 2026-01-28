# api/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# -------------------------
# Common
# -------------------------
class HealthResponse(BaseModel):
    ok: bool = True


# -------------------------
# ASR
# -------------------------
class AsrCreateResponse(BaseModel):
    job_id: str


class AsrJobResponse(BaseModel):
    job_id: str
    status: str
    transcript: Optional[str] = None
    error: Optional[str] = None


# -------------------------
# Quiz
# -------------------------
class QuizRequest(BaseModel):
    transcript: str = Field(..., description="ASR 결과 텍스트 또는 YouTube 자막 텍스트")
    level: Optional[str] = Field(default=None, description="예: A1/A2/B1 등")


class QuizResponse(BaseModel):
    quiz: Dict[str, Any]


# -------------------------
# Attempts / Grading
# -------------------------
class AttemptRequest(BaseModel):
    quiz: Dict[str, Any]
    user_answers: Dict[str, Any]


class AttemptResponse(BaseModel):
    score: Optional[float] = None
    wrong_items: List[Any] = Field(default_factory=list)
    feedback: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# YouTube Transcript / Quiz
# -------------------------
class YtTranscriptRequest(BaseModel):
    url: str
    lang: Optional[str] = "id"


class YtTranscriptResponse(BaseModel):
    video_id: str
    transcript: str


class YtQuizRequest(BaseModel):
    url: str
    level: str = "A2"
    lang: Optional[str] = "id"


# -------------------------
# Repeat CTA (Similar Quiz)
# -------------------------
class RepeatSimilarRequest(BaseModel):
    transcript: str
    level: Optional[str] = None
    prev_quiz: Dict[str, Any]  # 기존 quiz payload(questions 포함)
