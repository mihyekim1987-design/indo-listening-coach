# app.py
# -*- coding: utf-8 -*-
"""
인도네시아어 학습 도구 - Domain-Specific Learning Tool
사용자가 오디오, YouTube 링크, 또는 텍스트 링크를 제공하면
교육적 가치를 분석하고 퀴즈를 생성하여 학습을 돕는 Streamlit 앱
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import re
import glob
from datetime import datetime, timedelta
import streamlit as st
import torch
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import streamlit.components.v1 as components
import asyncio
import edge_tts
import hashlib
from pathlib import Path
import base64

from ui.header import inject_global_css, render_header
from ui.mode_state import (
    MODES,
    get_mode_state,
    reset_mode_ephemeral,
    record_mode_result,
)
from ui.effects import celebrate_confetti

# core 로직 모듈 (UI와 분리된 순수 파이썬 로직)
from core.asr import transcribe_audio as core_transcribe_audio
from core.level import get_level_profile
from core.quiz import generate_quiz as core_generate_quiz
from core.grading import grade_quiz

# OpenAI 설정
# 로컬 개발에서는 .env가 있으면 읽고, 배포에서는 무시되어도 문제 없음
load_dotenv()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY가 설정되지 않았습니다. .streamlit/secrets.toml 또는 환경변수를 확인하세요."
    )
    st.stop()


@st.cache_resource
def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


client = get_openai_client(OPENAI_API_KEY)

# Pydantic 모델 (Structured Outputs용)
from pydantic import BaseModel
from typing import Dict, List, Optional

# 프롬프트 불러오기
import prompts as P

missing = [
    name
    for name in [
        "QUIZ_PROMPT",
        "COACH_PROMPT",
        "EDUCATIONAL_ANALYSIS_PROMPT",
        "AI_LEARNING_COACH_PROMPT",
        "YOUTUBE_WRITING_FEEDBACK_PROMPT",
    ]
    if not hasattr(P, name)
]
if missing:
    raise ImportError(
        f"[prompts import check] Missing: {missing}\n"
        f"Loaded prompts.py from: {getattr(P, '__file__', 'unknown')}\n"
        f"Available names: {sorted([n for n in dir(P) if 'PROMPT' in n or 'CEFR' in n])}"
    )

QUIZ_PROMPT = P.QUIZ_PROMPT
COACH_PROMPT = P.COACH_PROMPT
EDUCATIONAL_ANALYSIS_PROMPT = P.EDUCATIONAL_ANALYSIS_PROMPT
AI_LEARNING_COACH_PROMPT = P.AI_LEARNING_COACH_PROMPT
YOUTUBE_WRITING_FEEDBACK_PROMPT = P.YOUTUBE_WRITING_FEEDBACK_PROMPT

# 상수 정의
APP_TITLE = "Bisa Q"  # 브라우저 탭(title)용
MODEL_ID = "Sparkplugx1904/whisper-base-id"
TARGET_SR = 16000


@st.cache_resource
def get_cached_asr_pipe(model_id: str):
    # transformers.pipeline은 이미 상단에서 import 되어 있으므로 그대로 사용
    return pipeline("automatic-speech-recognition", model=model_id)


asr_pipe = get_cached_asr_pipe(MODEL_ID)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOGO_PATH = Path("assets/Logo.png")

# =====================================================
# UI Labels (single source of truth)
# =====================================================

APP_TITLE = "Bisa Q"  # 브라우저 탭 제목(짧게)

UI = {
    "nav_listening": "Listening",
    "nav_watching": "Watching",
    "nav_reading": "Reading",
    "nav_progress": "Progress",
    "nav_settings": "Settings",
    "cta_make_it_bisa": "Make it Bisa",
    "home_title": "Don't Just Listen.<br>Speak Indonesian.",
    "home_subtext": "Turn your favorite videos and podcasts<br>into interactive AI lessons.<br>Move from passive input<br>to active fluency with Bisa Q.",
    "home_cta_prompt": "Make it Bisa",
    "card_audio": "Listening",
    "card_youtube": "Watching",
    "card_text": "Reading",
    "card_results": "Progress",
    "card_settings": "Settings",
    "progress_header": "Progress",
    "tab_dashboard": "Dashboard",
    "tab_review": "Review",
    "tab_speaking": "Speaking",
    "tab_quiz": "Quiz",
    "tab_memory": "Memory",
    "btn_home": "Home",
    "btn_view_progress": "View Progress",
}

# TTS 캐시 디렉토리
TTS_CACHE_DIR = os.path.join(LOG_DIR, "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# 샘플 링크
SAMPLE_LINKS = {
    "Wikisource": "https://id.wikisource.org/wiki/Pangeran_Yang_Bahagia",
    "Wikinews (VOA)": "https://www.voaindonesia.com/a/guru-kreator-konten-mengapa-perlu-/7972528.html",
    "VOA Indonesia": "https://www.voaindonesia.com/a/guru-indonesia-dan-persepsi-tentang-as/6920962.html",
}

# =====================================================
# CEFR 기반 취약 카테고리 맵핑
# =====================================================

CEFR_CATEGORIES = {
    "vocabulary": {
        "name": "어휘 (Kosakata)",
        "icon": "📚",
        "description": "인도네시아어 단어의 의미와 사용법을 이해하는 능력입니다. 기본 단어부터 격식 어휘, 접두사/접미사까지 포함합니다.",
        "keywords": ["arti", "makna", "kata", "kosakata", "어휘", "단어", "뜻", "의미"],
        "subcategories": {
            "basic_words": "기본 단어",
            "numbers": "숫자/수량 표현",
            "time_expressions": "시간 표현",
            "formal_vocabulary": "격식 어휘",
            "idioms": "관용구/숙어",
            "affixes": "접두사/접미사 (me-, ber-, -kan, -i)",
        },
    },
    "grammar": {
        "name": "문법 (Tata Bahasa)",
        "icon": "📝",
        "description": "인도네시아어 문법 구조와 접사 체계를 이해하는 능력입니다. 수동태, 사역형, 시제 표현 등이 포함됩니다.",
        "keywords": [
            "di-",
            "ter-",
            "me-",
            "ber-",
            "-kan",
            "접사",
            "수동",
            "문법",
            "시제",
        ],
        "subcategories": {
            "tense": "시제 (sudah, akan, sedang)",
            "passive": "수동태 (di-, ter-)",
            "causative": "사역형 (-kan)",
            "prefix_suffix": "접사 체계",
            "reduplication": "반복어 (reduplikasi)",
            "conjunctions": "접속사",
            "prepositions": "전치사 (di, ke, dari)",
        },
    },
    "politeness": {
        "name": "경어/존칭 (Kesopanan)",
        "icon": "🎩",
        "description": "격식체, 존칭, 요청 표현 등 인도네시아어의 예의 표현을 이해하고 사용하는 능력입니다.",
        "keywords": ["bapak", "ibu", "pak", "bu", "존칭", "경어", "tolong", "mohon"],
        "subcategories": {
            "formal_register": "격식체",
            "honorifics": "존칭 (Bapak, Ibu, Pak, Bu)",
            "humble_forms": "겸양어",
            "request_forms": "요청 표현 (tolong, mohon)",
        },
    },
    "comprehension": {
        "name": "독해/이해 (Pemahaman)",
        "icon": "🔍",
        "description": "텍스트의 중심 내용, 세부 정보를 파악하고 문맥을 통해 추론하는 능력입니다.",
        "keywords": ["utama", "pokok", "중심", "주제", "내용", "이해"],
        "subcategories": {
            "main_idea": "중심 내용 파악",
            "detail": "세부 정보",
            "inference": "추론",
            "context": "문맥 파악",
        },
    },
    "numbers": {
        "name": "숫자/수량 (Angka)",
        "icon": "🔢",
        "description": "인도네시아어의 기수, 서수, 수량 표현을 이해하고 사용하는 능력입니다.",
        "keywords": ["berapa", "jumlah", "angka", "숫자", "몇", "수량"],
        "subcategories": {
            "cardinal": "기수",
            "ordinal": "서수",
            "quantity": "수량 표현",
        },
    },
    "time": {
        "name": "시간 표현 (Waktu)",
        "icon": "⏰",
        "description": "시계 시간, 날짜, 기간 등 시간과 관련된 표현을 이해하고 사용하는 능력입니다.",
        "keywords": ["kapan", "waktu", "tanggal", "jam", "시간", "날짜", "언제"],
        "subcategories": {
            "clock_time": "시계 시간",
            "date": "날짜",
            "duration": "기간",
        },
    },
}

# CEFR 레벨별 설명
CEFR_LEVEL_DESCRIPTORS = {
    "A1": {
        "description": "입문 - 기본적인 표현과 문장 이해",
        "focus_categories": ["vocabulary", "numbers", "time"],
        "expected_accuracy": 70,
    },
    "A2": {
        "description": "초급 - 일상적인 표현과 기본 대화",
        "focus_categories": ["vocabulary", "grammar", "comprehension"],
        "expected_accuracy": 65,
    },
    "B1": {
        "description": "중급 - 일반적인 주제 이해 및 표현",
        "focus_categories": ["grammar", "politeness", "comprehension"],
        "expected_accuracy": 60,
    },
    "B2": {
        "description": "중상급 - 복잡한 텍스트 이해, 유창한 대화",
        "focus_categories": ["grammar", "comprehension", "politeness"],
        "expected_accuracy": 55,
    },
}

# Spaced Repetition 간격 (일 단위) - SM-2 알고리즘 기반
SRS_INTERVALS = [1, 3, 7, 14, 30, 60, 120]
SRS_DATA_FILE = os.path.join(LOG_DIR, "spaced_repetition_data.json")
LEARNING_HISTORY_FILE = os.path.join(LOG_DIR, "learning_history.json")

# TTS 속도 옵션
TTS_SPEED_OPTIONS = {
    "very_slow": {"label": "매우 느리게 (0.3x)", "rate": 0.3},
    "slow": {"label": "느리게 (0.5x)", "rate": 0.5},
    "normal": {"label": "보통 (1.0x)", "rate": 1.0},
    "fast": {"label": "빠르게 (1.5x)", "rate": 1.5},
}

# =====================================================
# 1. ASR (Automatic Speech Recognition) 기능
# =====================================================


@st.cache_resource
def load_asr():
    """
    Whisper ASR 모델을 로드합니다.
    @st.cache_resource 데코레이터로 한 번만 로드됩니다.
    """
    try:
        # CPU 스레드 수 제한 (과도한 스레드로 인한 멈춤 방지)
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    except Exception:
        pass

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        device=device,
    )


# =====================================================
# 2. 텍스트 추출 기능 (웹 크롤링)
# =====================================================


def format_text_readable(text: str, lines_per_paragraph: int = 5) -> str:
    """
    텍스트를 가독성 좋게 포맷팅합니다.
    일정 줄 수마다 문단을 나눕니다

    Args:
        text: 원본 텍스트
        lines_per_paragraph: 문단당 줄 수 (기본값: 5)

    Returns:
        str: 포맷팅된 텍스트
    """
    if not text:
        return text

    # 줄 단위로 분리
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return text

    # 문단으로 그룹화
    paragraphs = []
    current_paragraph = []

    for i, line in enumerate(lines):
        current_paragraph.append(line)

        # 일정 줄 수마다 또는 마지막 줄일 때 문단 구분
        if (i + 1) % lines_per_paragraph == 0 or (i + 1) == len(lines):
            # 현재 문단을 공백으로 연결
            paragraph_text = " ".join(current_paragraph)
            paragraphs.append(paragraph_text)
            current_paragraph = []

    # 문단들을 빈 줄로 구분하여 결합
    formatted_text = "\n\n".join(paragraphs)

    return formatted_text


def format_audio_transcript(text: str, sentences_per_paragraph: int = 3) -> str:
    """
    오디오 ASR 결과를 가독성 좋게 포맷팅합니다.
    문장 단위로 나누고, 일정 문장 수마다 문단을 구분합니다.

    Args:
        text: 원본 ASR 텍스트 (보통 줄바꿈 없는 긴 텍스트)
        sentences_per_paragraph: 문단당 문장 수 (기본값: 3)

    Returns:
        str: 포맷팅된 텍스트
    """
    if not text:
        return text

    # 문장 단위로 분리 (마침표, 물음표, 느낌표 기준)
    # 인도네시아어에서도 같은 구두점 사용
    import re

    # 문장 분리 패턴: . ! ? 뒤에 공백이나 끝이 오는 경우
    sentence_pattern = r"([^.!?]+[.!?]+)"
    sentences = re.findall(sentence_pattern, text)

    # 패턴에 매칭되지 않은 나머지 텍스트 처리
    remaining = re.sub(sentence_pattern, "", text).strip()
    if remaining:
        sentences.append(remaining)

    # 문장이 없으면 원본 반환
    if not sentences:
        return text

    # 문장들을 정리 (앞뒤 공백 제거)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 문단으로 그룹화
    paragraphs = []
    current_paragraph = []

    for i, sentence in enumerate(sentences):
        current_paragraph.append(sentence)

        # 일정 문장 수마다 또는 마지막 문장일 때 문단 구분
        if (i + 1) % sentences_per_paragraph == 0 or (i + 1) == len(sentences):
            # 현재 문단을 공백으로 연결
            paragraph_text = " ".join(current_paragraph)
            paragraphs.append(paragraph_text)
            current_paragraph = []

    # 문단들을 빈 줄로 구분하여 결합
    formatted_text = "\n\n".join(paragraphs)

    return formatted_text


def normalize_url_input(raw: str) -> str:
    """
    사용자가 붙여넣은 값이
    - 순수 URL
    - [텍스트](URL) 마크다운 링크
    - <URL> 형태
    - 앞뒤 공백/개행 포함
    일 때 URL만 안전하게 추출해 반환
    """
    if not raw:
        return ""

    s = raw.strip()

    # <https://...> 형태 제거
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()

    # [title](url) 형태면 url 부분만 뽑기
    m = re.search(r"\((https?://[^)]+)\)", s)
    if m:
        return m.group(1).strip()

    # 혹시 "https://... ](https://...)" 같이 섞였으면 첫 URL만 추출
    m2 = re.search(r"(https?://\S+)", s)
    if m2:
        return m2.group(1).rstrip(")").strip()

    return s


def extract_text_from_url(url: str, timeout: int = 20, min_chars: int = 200) -> dict:
    """
    웹 URL에서 텍스트(본문)와 제목을 추출합니다.
    - 확장자(.html 등)로 차단하지 않음
    - User-Agent 포함, redirect 허용
    - HTML 파싱 후 article/main 우선, 없으면 가장 긴 본문 후보 선택
    """
    import re

    def _normalize_text(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # 줄 끝 공백 제거
        s = "\n".join(line.strip() for line in s.split("\n"))
        # 3줄 이상 빈 줄 -> 2줄로 축소
        s = re.sub(r"\n{3,}", "\n\n", s)
        # 연속 공백 축소(문장 내부는 보존)
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    try:
        if not url or not isinstance(url, str):
            return {
                "success": False,
                "title": "",
                "text": "",
                "error": "URL이 비어있습니다.",
            }

        url = url.strip()

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "id,en;q=0.8,ko;q=0.6",
            "Cache-Control": "no-cache",
        }

        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        content_type = (resp.headers.get("Content-Type") or "").lower()

        # 인코딩 보정 (특히 iso-8859-1로 잘못 잡히는 케이스 방지)
        if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "latin-1"):
            resp.encoding = resp.apparent_encoding or "utf-8"

        html = resp.text or ""
        if not html.strip():
            return {
                "success": False,
                "title": "",
                "text": "",
                "error": "응답 본문이 비어있습니다(차단/권한/네트워크 가능).",
            }

        # HTML이 아닌 것 같으면 실패 처리 (단, 일부 서버는 content-type이 이상할 수 있어 html 첫 글자도 체크)
        if ("text/html" not in content_type) and (
            "<html" not in html.lower() and "<body" not in html.lower()
        ):
            return {
                "success": False,
                "title": "",
                "text": "",
                "error": f"HTML 페이지가 아닌 것 같습니다. (Content-Type: {content_type})",
            }

        # 파서: lxml 우선, 없으면 html.parser
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        # 불필요 태그 제거
        for tag in soup(
            ["script", "style", "noscript", "svg", "canvas", "iframe", "form"]
        ):
            tag.decompose()
        for tag in soup.select(
            "nav, header, footer, aside, .nav, .menu, .footer, .header, .sidebar, .advert, .ads"
        ):
            tag.decompose()

        # 제목 추출
        title = ""
        if soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(strip=True)
        # og:title 우선 적용(있으면 더 정확)
        og_title = soup.select_one('meta[property="og:title"]')
        if og_title and og_title.get("content"):
            title = og_title.get("content").strip()

        # 본문 후보 우선순위: article > main > (role=main) > body
        candidates = []

        def _get_text_from_node(node) -> str:
            if not node:
                return ""
            txt = node.get_text(separator="\n", strip=True)
            return _normalize_text(txt)

        for selector in [
            "article",
            "main",
            '[role="main"]',
            ".article",
            ".content",
            ".post",
            ".entry-content",
        ]:
            node = soup.select_one(selector)
            txt = _get_text_from_node(node)
            if txt and len(txt) >= 50:
                candidates.append(txt)

        # 후보가 없으면: 가장 텍스트가 긴 컨테이너를 선택
        if not candidates:
            # 너무 많이 돌면 느려져서 상위 일부만
            containers = soup.find_all(["article", "main", "section", "div"], limit=300)
            best_txt = ""
            for c in containers:
                txt = _get_text_from_node(c)
                # 너무 짧은 건 제외
                if len(txt) > len(best_txt) and len(txt) >= 80:
                    best_txt = txt
            if best_txt:
                candidates.append(best_txt)

        # 마지막 fallback: body 전체
        if not candidates:
            body_txt = _get_text_from_node(soup.body)
            if body_txt:
                candidates.append(body_txt)

        text = max(candidates, key=len, default="")
        text = _normalize_text(text)

        # 너무 짧으면 실패로 처리(“추출은 됐지만 내용이 없어서 학습 불가” 방지)
        if len(text) < min_chars:
            return {
                "success": False,
                "title": title or "",
                "text": "",
                "error": (
                    f"추출된 텍스트가 너무 짧습니다({len(text)}자). "
                    "해당 페이지가 JS 렌더링/로그인 필요/차단/본문이 이미지일 수 있습니다."
                ),
            }

        return {
            "success": True,
            "title": title or "Untitled",
            "text": text,
            "error": "",
        }

    except requests.exceptions.RequestException as e:
        return {"success": False, "title": "", "text": "", "error": f"요청 실패: {e}"}
    except Exception as e:
        return {
            "success": False,
            "title": "",
            "text": "",
            "error": f"추출 중 오류: {e}",
        }


def extract_youtube_id(url: str) -> str:
    """
    YouTube URL에서 비디오 ID를 추출합니다.

    Args:
        url: YouTube URL

    Returns:
        str: 비디오 ID (실패 시 빈 문자열)
    """
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com\/embed\/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return ""


def format_transcript_readable(fetched) -> str:
    """
    자막을 가독성 좋게 포맷팅합니다.
    시간 기준으로 문단을 나누고, 문장 간 띄어쓰기를 개선합니다.

    Args:
        fetched: FetchedTranscript 객체

    Returns:
        str: 포맷팅된 자막 텍스트
    """
    formatted_lines = []
    current_paragraph = []
    last_time = 0

    for snippet in fetched:
        text = snippet.text.strip()
        if not text:
            continue

        # 시간 정보 (snippet.start는 초 단위)
        current_time = snippet.start

        # 30초마다 문단 나누기
        if current_time - last_time > 30 and current_paragraph:
            # 현재 문단을 한 줄로 합치고 저장
            paragraph_text = " ".join(current_paragraph)
            formatted_lines.append(paragraph_text)
            formatted_lines.append("")  # 빈 줄 추가 (문단 구분)
            current_paragraph = []
            last_time = current_time

        current_paragraph.append(text)

    # 마지막 문단 추가
    if current_paragraph:
        paragraph_text = " ".join(current_paragraph)
        formatted_lines.append(paragraph_text)

    # 전체 텍스트 조합
    full_text = "\n".join(formatted_lines)

    # 연속된 빈 줄을 하나로 정리
    while "\n\n\n" in full_text:
        full_text = full_text.replace("\n\n\n", "\n\n")

    return full_text.strip()


@st.cache_data(ttl=3600, show_spinner=False)  # 1시간 캐싱 (video_id가 캐시 키)
def get_youtube_transcript(video_id: str, language: str = "id") -> dict:
    """
    YouTube 비디오의 자막을 가져옵니다. (캐싱됨)

    Args:
        video_id: YouTube 비디오 ID
        language: 원하는 언어 코드 (기본값: "id" - 인도네시아어)

    Returns:
        dict: {"success": bool, "transcript": str, "error": str, "language_used": str}
    """
    try:
        # YouTubeTranscriptApi 인스턴스 생성
        api = YouTubeTranscriptApi()

        # 먼저 원하는 언어(인도네시아어)로 자막 가져오기 시도
        try:
            # fetch 메서드 사용 (인스턴스 메서드)
            fetched = api.fetch(video_id, languages=[language])

            # FetchedTranscript 객체에서 텍스트 추출 및 포맷팅
            formatted_text = format_transcript_readable(fetched)

            return {
                "success": True,
                "transcript": formatted_text,
                "language_used": language,
                "error": None,
            }

        except NoTranscriptFound:
            # 인도네시아어 자막이 없으면 영어로 시도
            try:
                fetched = api.fetch(video_id, languages=["en"])
                formatted_text = format_transcript_readable(fetched)

                return {
                    "success": True,
                    "transcript": formatted_text,
                    "language_used": "en (영어 자막)",
                    "error": None,
                }
            except NoTranscriptFound:
                # 사용 가능한 자막 목록 가져오기
                try:
                    transcript_list = api.list(video_id)
                    available = [t.language_code for t in transcript_list]
                    return {
                        "success": False,
                        "transcript": "",
                        "language_used": "",
                        "error": f"인도네시아어/영어 자막이 없습니다. 사용 가능한 언어: {', '.join(available)}",
                    }
                except:
                    return {
                        "success": False,
                        "transcript": "",
                        "language_used": "",
                        "error": "인도네시아어 자막을 찾을 수 없습니다.",
                    }

    except TranscriptsDisabled:
        return {
            "success": False,
            "transcript": "",
            "language_used": "",
            "error": "이 영상은 자막이 비활성화되어 있습니다.",
        }
    except Exception as e:
        return {
            "success": False,
            "transcript": "",
            "language_used": "",
            "error": f"자막 가져오기 실패: {str(e)}",
        }


def show_confetti():
    """
    복습 퀴즈 완료 시 축하 confetti 효과를 보여줍니다.
    """
    components.html(
        """
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <script>
            // 3초 동안 화려한 폭죽 효과
            var duration = 3 * 1000;
            var end = Date.now() + duration;

            (function frame() {
                confetti({
                    particleCount: 100,
                    startVelocity: 30,
                    spread: 360,
                    origin: {
                        x: Math.random(),
                        y: Math.random() - 0.2
                    }
                });

                if (Date.now() < end) {
                    requestAnimationFrame(frame);
                }
            }());
        </script>
        """,
        height=200,
        scrolling=False,
    )


def reset_learning_state(source_type: str, source_id: str = None):
    """
    학습 상태를 초기화합니다. source_id가 변경되면 이전 데이터를 모두 제거합니다.

    Args:
        source_type: "audio", "youtube", "text" 중 하나
        source_id: 소스 식별자 (youtube의 경우 video_id, 없으면 전체 초기화)
    """
    if source_type == "audio":
        keys_to_remove = [
            "audio_transcript",
            "audio_quiz",
            "audio_coach",
            "start_audio_quiz_generation",
        ]
    elif source_type == "youtube":
        keys_to_remove = [
            "youtube_transcript",
            "youtube_quiz",
            "youtube_coach",
            "start_quiz_generation",
            "youtube_quiz_video_id",
            "youtube_current_url",
        ]
    elif source_type == "text":
        keys_to_remove = [
            "extracted_text",
            "extracted_title",
            "text_quiz",
            "text_coach",
            "start_text_quiz_generation",
        ]
    else:
        return

    # source_id가 제공되고, 현재 저장된 ID와 다를 때만 초기화
    if source_id and source_type == "youtube":
        current_id = st.session_state.get("youtube_quiz_video_id", "")
        if current_id == source_id:
            # 같은 소스면 초기화하지 않음
            return

    # 키 제거
    for key in keys_to_remove:
        st.session_state.pop(key, None)


# =====================================================
# 3. LLM 호출 (OpenAI API)
# =====================================================


def safe_prompt_fill(template: str, **kwargs) -> str:
    """
    안전한 프롬프트 치환 함수.
    - str.format()을 쓰지 않아서, 프롬프트 내부의 JSON 예시 { } 때문에 KeyError가 나는 문제를 방지합니다.
    - {key} 형태로 들어있는 것만 치환합니다.
    """
    out = template
    for k, v in kwargs.items():
        token = "{" + k + "}"
        out = out.replace(token, "" if v is None else str(v))
    return out


def llm_json(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """
    OpenAI API를 호출하여 JSON 형식의 응답을 받습니다.

    Args:
        prompt: 프롬프트 텍스트
        model: 사용할 모델 (기본값: gpt-4o-mini)

    Returns:
        dict: JSON 파싱된 응답
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You must output ONLY valid JSON. No extra text, no markdown, no code blocks.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},  # JSON mode 활성화
    )
    text = resp.choices[0].message.content.strip()

    # 디버그용: 원문 응답 저장
    if "last_llm_response" not in st.session_state:
        st.session_state["last_llm_response"] = {}
    st.session_state["last_llm_response"]["raw_text"] = text

    # JSON 정리: 마크다운 코드 블록 제거
    cleaned_text = text

    # ```json ... ``` 형태의 코드 블록 제거
    if "```json" in cleaned_text:
        cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned_text:
        # ``` ... ``` 형태도 처리
        parts = cleaned_text.split("```")
        if len(parts) >= 3:
            cleaned_text = parts[1].strip()

    # 앞뒤 공백 제거
    cleaned_text = cleaned_text.strip()

    # JSON 파싱 시도
    try:
        parsed = json.loads(cleaned_text)
        st.session_state["last_llm_response"]["parsed"] = parsed
        st.session_state["last_llm_response"]["cleaned_text"] = cleaned_text
        return parsed
    except json.JSONDecodeError as e:
        st.session_state["last_llm_response"]["error"] = str(e)
        st.session_state["last_llm_response"]["cleaned_text"] = cleaned_text

        # 에러 표시
        st.error(f"❌ JSON 파싱 실패: {str(e)}")

        with st.expander("🔍 원본 응답 확인 (디버그)"):
            st.markdown("**원본 응답:**")
            st.code(text, language="text")
            st.markdown("**정리된 텍스트:**")
            st.code(cleaned_text, language="json")
            st.markdown("**파싱 에러:**")
            st.code(str(e))

        st.warning(
            "💡 퀴즈 생성을 다시 시도해주세요. 문제가 계속되면 텍스트 길이를 줄이거나 다른 자료를 사용해보세요."
        )
        raise


# =====================================================
# 3-2. Structured Outputs (Pydantic 모델)
# =====================================================


class ChoiceNotes(BaseModel):
    """각 선택지별 해설"""

    A: str
    B: str
    C: str
    D: str


class TomorrowPlanStep(BaseModel):
    """학습 플랜의 각 단계"""

    minute: str
    task: str


class ShadowingSentence(BaseModel):
    """Shadowing 연습 문장"""

    id: str  # 인도네시아어 문장
    ko: str  # 한국어 번역


class ExplainItem(BaseModel):
    """각 문항별 해설"""

    id: int
    is_correct: bool
    correct_explain_ko: str
    wrong_reason_ko: str
    summary_ko: str = ""
    choice_notes_ko: ChoiceNotes
    evidence_quote: str


class CoachResponse(BaseModel):
    """채점 및 코칭 전체 응답"""

    items: List[ExplainItem]
    weak_points_ko: List[str]
    tomorrow_plan_10min_ko: List[TomorrowPlanStep]
    shadowing_sentences: List[ShadowingSentence]


def llm_structured(prompt: str, response_model, model: str = "gpt-4o-mini"):
    """
    OpenAI Structured Outputs를 사용하여 스키마에 맞는 응답을 받습니다.

    Args:
        prompt: 프롬프트 텍스트
        response_model: Pydantic BaseModel 클래스
        model: 사용할 모델

    Returns:
        dict: Pydantic 모델을 딕셔너리로 변환한 결과
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an Indonesian language learning coach. Return structured output that matches the schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=response_model,
            temperature=0.3,
        )

        parsed_response = completion.choices[0].message.parsed

        # None 체크
        if parsed_response is None:
            raise ValueError("LLM returned empty response (parsed is None)")

        # 디버그용: 원문 응답 저장
        if "last_llm_response" not in st.session_state:
            st.session_state["last_llm_response"] = {}
        st.session_state["last_llm_response"]["parsed"] = parsed_response.model_dump()
        st.session_state["last_llm_response"]["model"] = response_model.__name__

        return parsed_response.model_dump()

    except Exception as e:
        # 디버그용: 에러 저장
        if "last_llm_response" not in st.session_state:
            st.session_state["last_llm_response"] = {}
        st.session_state["last_llm_response"]["error"] = str(e)
        raise


# =====================================================
# 3-3. 취약점 분석 시스템
# =====================================================


class WeaknessAnalyzer:
    """CEFR 기반 취약 카테고리 분석기"""

    @staticmethod
    def categorize_question(question: dict) -> tuple:
        """문제를 카테고리로 분류"""
        q_text = (
            question.get("question", "") + " " + str(question.get("choices", {}))
        ).lower()

        # 키워드 기반 카테고리 매칭
        for cat_key, cat_info in CEFR_CATEGORIES.items():
            keywords = cat_info.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in q_text:
                    # 서브카테고리 결정
                    subcategories = list(cat_info.get("subcategories", {}).keys())
                    subcategory = subcategories[0] if subcategories else "general"
                    return cat_key, subcategory

        # 기본값: comprehension
        return "comprehension", "detail"

    @staticmethod
    def analyze_wrong_answer(
        question: dict, user_answer: str, correct_answer: str
    ) -> dict:
        """오답 분석하여 취약 카테고리 판단"""
        category, subcategory = WeaknessAnalyzer.categorize_question(question)

        return {
            "question_id": question.get("id"),
            "question": question.get("question", ""),
            "category": category,
            "subcategory": subcategory,
            "evidence_quote": question.get("evidence_quote", ""),
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "choices": question.get("choices", {}),
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def get_weakness_summary(wrong_items: list) -> dict:
        """오답 목록에서 취약점 요약 생성"""
        category_counts = {}
        subcategory_counts = {}
        evidence_quotes = []

        for item in wrong_items:
            cat = item.get("category", "comprehension")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            subcat = f"{cat}.{item.get('subcategory', 'general')}"
            subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

            if item.get("evidence_quote"):
                evidence_quotes.append(
                    {
                        "text": item["evidence_quote"],
                        "category": cat,
                        "question_id": item.get("question_id"),
                    }
                )

        # 가장 취약한 카테고리
        primary_weakness = (
            max(category_counts, key=category_counts.get) if category_counts else None
        )

        # 추천 학습 활동 생성
        recommendations = []
        for cat, count in category_counts.items():
            if count >= 1:
                cat_info = CEFR_CATEGORIES.get(cat, {})
                recommendations.append(
                    {
                        "category": cat,
                        "icon": cat_info.get("icon", "📌"),
                        "name": cat_info.get("name", cat),
                        "count": count,
                        "message": f"{cat_info.get('name', cat)} 영역에서 {count}개 오답",
                        "activity": WeaknessAnalyzer._get_activity_recommendation(cat),
                    }
                )

        # count 순으로 정렬
        recommendations.sort(key=lambda x: x["count"], reverse=True)

        return {
            "total_wrong": len(wrong_items),
            "category_breakdown": category_counts,
            "subcategory_breakdown": subcategory_counts,
            "primary_weakness": primary_weakness,
            "evidence_quotes": evidence_quotes,
            "recommendations": recommendations,
        }

    @staticmethod
    def _get_activity_recommendation(category: str) -> str:
        """카테고리별 학습 활동 추천"""
        activities = {
            "vocabulary": "플래시카드로 새 단어 20개 암기 + 예문 작성",
            "grammar": "접두사/접미사 패턴 표 만들기 + 변형 연습",
            "politeness": "상황별 경어 표현 대화문 만들기",
            "comprehension": "짧은 기사 읽고 요약문 작성하기",
            "numbers": "인도네시아어 숫자 1-100 빠르게 읽기 연습",
            "time": "일정표를 인도네시아어로 작성해보기",
        }
        return activities.get(category, "관련 예문 5개 필사하기")


# =====================================================
# 3-4. 반복 학습 시스템 (틀린 문제 정답까지)
# =====================================================


class RepeatLearningManager:
    """반복 학습 관리자 - 틀린 문제를 정답까지 반복"""

    @staticmethod
    def _default_state() -> dict:
        return {
            "wrong_queue": [],  # 틀린 문제 대기열
            "current_question": None,  # 현재 풀고 있는 문제
            "retry_count": {},  # 문제별 재시도 횟수
            "completed": [],  # 완료된 문제
            "total_retries": 0,  # 총 시도 횟수
            "active": False,  # 반복 학습 모드 활성화
            "celebration_shown": False,
            "kind": None,  # wrong | similar
        }

    @classmethod
    def init_state(cls, mode: str) -> dict:
        """세션 상태 초기화"""
        mode_state = get_mode_state(mode)
        if "repeat_learning" not in mode_state:
            mode_state["repeat_learning"] = cls._default_state()
        return mode_state["repeat_learning"]

    @classmethod
    def get_state(cls, mode: str) -> dict:
        return cls.init_state(mode)

    @classmethod
    def start_repeat_learning(cls, mode: str, wrong_items: list, quiz_questions: list):
        """반복 학습 시작"""
        state = cls.init_state(mode)
        mode_state = get_mode_state(mode)

        # 초기화
        state["wrong_queue"] = []
        state["completed"] = []
        state["retry_count"] = {}
        state["total_retries"] = 0
        state["active"] = True
        state["celebration_shown"] = False
        state["kind"] = "wrong"
        mode_state["retry_active"] = True
        mode_state["retry_completed"] = False
        mode_state["retry_mastered"] = False
        mode_state["retry_quiz"] = {"questions": quiz_questions}

        init_shadowing_review_bank(mode, wrong_items, quiz_questions)

        # quiz_questions를 딕셔너리로 변환
        q_dict = {str(q.get("id")): q for q in quiz_questions}

        for item in wrong_items:
            # id 또는 question_id 필드 확인
            q_id = str(item.get("id") or item.get("question_id", ""))
            full_question = q_dict.get(q_id, {})

            # 원본 문제 정보에 오답 정보 추가
            question_data = {
                **full_question,
                "id": q_id,  # ID 명시적으로 설정
                "user_wrong_answer": item.get("user_answer", ""),
                "evidence_quote": item.get(
                    "evidence_quote", full_question.get("evidence_quote", "")
                ),
                "why_correct_ko": item.get("why_correct_ko", ""),
                "why_user_wrong_ko": item.get("why_user_wrong_ko", ""),
                "category": item.get("category", "comprehension"),
            }

            state["wrong_queue"].append(question_data)
            state["retry_count"][q_id] = 0

    @classmethod
    def start_custom(cls, mode: str, questions: list, kind: str):
        """제공된 문제 리스트로 반복 학습 시작"""
        state = cls.init_state(mode)
        mode_state = get_mode_state(mode)

        # 초기화
        state["wrong_queue"] = []
        state["completed"] = []
        state["retry_count"] = {}
        state["total_retries"] = 0
        state["active"] = True
        state["celebration_shown"] = False
        state["kind"] = kind
        mode_state["retry_active"] = True
        mode_state["retry_completed"] = False
        mode_state["retry_mastered"] = False
        mode_state["retry_quiz"] = {"questions": questions}

        seed_items = []
        for q in questions or []:
            q_id = str(q.get("id") or "")
            if not q_id:
                continue
            seed_items.append(
                {
                    "id": q_id,
                    "question_id": q_id,
                    "user_answer": "",
                    "correct_answer": q.get("answer", ""),
                    "evidence_quote": q.get("evidence_quote", ""),
                    "category": q.get("category", "comprehension"),
                }
            )

        init_shadowing_review_bank(mode, seed_items, questions or [])

        for q in questions or []:
            q_id = str(q.get("id") or "")
            if not q_id:
                continue
            question_data = {
                **q,
                "id": q_id,
            }
            state["wrong_queue"].append(question_data)
            state["retry_count"][q_id] = 0

    @classmethod
    def get_next_question(cls, mode: str) -> Optional[dict]:
        """다음 풀어야 할 문제 반환"""
        state = cls.init_state(mode)

        if state["wrong_queue"]:
            state["current_question"] = state["wrong_queue"][0]
            return state["current_question"]
        return None

    @classmethod
    def check_answer(cls, mode: str, user_answer: str) -> tuple:
        """답안 확인"""
        state = cls.init_state(mode)
        current = state["current_question"]

        if not current:
            return False, {"error": "현재 문제 없음"}

        q_id = str(current.get("id"))
        correct_answer = current.get("answer", "")

        state["retry_count"][q_id] = state["retry_count"].get(q_id, 0) + 1
        state["total_retries"] += 1

        is_correct = user_answer.strip().upper() == correct_answer.strip().upper()

        result = {
            "question_id": q_id,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "retry_count": state["retry_count"][q_id],
        }

        if is_correct:
            # 정답! 대기열에서 제거하고 완료 목록에 추가
            state["wrong_queue"] = [
                q for q in state["wrong_queue"] if str(q.get("id")) != q_id
            ]
            state["completed"].append(
                {**current, "retries_needed": state["retry_count"][q_id]}
            )
            state["current_question"] = None

        return is_correct, result

    @classmethod
    def replace_with_similar(cls, mode: str, similar_question: dict):
        """현재 문제를 유사 문제로 교체"""
        state = cls.init_state(mode)

        if state["wrong_queue"]:
            original_id = state["wrong_queue"][0].get("id")
            similar_question["original_id"] = original_id
            similar_question["is_similar"] = True
            state["wrong_queue"][0] = similar_question
            state["current_question"] = similar_question

    @classmethod
    def get_progress(cls, mode: str) -> dict:
        """진행 상황 반환"""
        state = cls.init_state(mode)

        total = len(state["completed"]) + len(state["wrong_queue"])
        completed = len(state["completed"])

        return {
            "total": total,
            "completed": completed,
            "remaining": len(state["wrong_queue"]),
            "total_retries": state["total_retries"],
            "progress_percent": int((completed / total) * 100) if total > 0 else 0,
            "active": state.get("active", False),
        }

    @classmethod
    def is_complete(cls, mode: str) -> bool:
        """모든 문제 완료 여부"""
        state = cls.init_state(mode)
        # active 상태이고, wrong_queue가 비어있고, completed가 있을 때만 완료
        return (
            state.get("active", False)
            and len(state["wrong_queue"]) == 0
            and len(state["completed"]) > 0
        )

    @classmethod
    def reset(cls, mode: str):
        """상태 초기화"""
        mode_state = get_mode_state(mode)
        mode_state["repeat_learning"] = cls._default_state()
        mode_state["retry_active"] = False
        mode_state["retry_completed"] = False
        mode_state["retry_mastered"] = False
        mode_state["retry_quiz"] = None


# =====================================================
# 3-5. Spaced Repetition 시스템 (SM-2 알고리즘)
# =====================================================


class SpacedRepetitionSystem:
    """간격 반복 학습 시스템"""

    @staticmethod
    def _load_data() -> dict:
        """저장된 SRS 데이터 로드"""
        if os.path.exists(SRS_DATA_FILE):
            try:
                with open(SRS_DATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"items": {}, "stats": {"total_reviews": 0}}

    @staticmethod
    def _save_data(data: dict):
        """SRS 데이터 저장"""
        with open(SRS_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def add_item(cls, item_id: str, category: str, content: dict):
        """새 학습 항목 추가"""
        data = cls._load_data()

        if item_id not in data["items"]:
            data["items"][item_id] = {
                "id": item_id,
                "category": category,
                "content": content,
                "level": 0,  # 복습 레벨 (0-6)
                "ease_factor": 2.5,  # 난이도 계수
                "next_review": datetime.now().isoformat(),
                "last_review": None,
                "review_count": 0,
                "correct_count": 0,
                "created_at": datetime.now().isoformat(),
            }
            cls._save_data(data)

    @classmethod
    def record_review(cls, item_id: str, is_correct: bool, quality: int = 3):
        """
        복습 결과 기록 (SM-2 알고리즘)
        quality: 0-5 (0=완전 모름, 5=완벽)
        """
        data = cls._load_data()

        if item_id not in data["items"]:
            return

        item = data["items"][item_id]
        item["review_count"] += 1
        item["last_review"] = datetime.now().isoformat()

        if is_correct and quality >= 3:
            item["correct_count"] += 1

            # SM-2 간격 계산
            if item["level"] == 0:
                interval = 1
            elif item["level"] == 1:
                interval = 3
            else:
                interval = SRS_INTERVALS[min(item["level"], len(SRS_INTERVALS) - 1)]

            # Ease factor 조정
            item["ease_factor"] = max(
                1.3,
                item["ease_factor"]
                + 0.1
                - (5 - quality) * (0.08 + (5 - quality) * 0.02),
            )
            interval = int(interval * item["ease_factor"])

            item["level"] = min(item["level"] + 1, len(SRS_INTERVALS) - 1)
        else:
            # 오답 또는 품질 낮음: 레벨 리셋
            item["level"] = 0
            interval = 1

        item["next_review"] = (datetime.now() + timedelta(days=interval)).isoformat()
        data["stats"]["total_reviews"] += 1
        cls._save_data(data)

    @classmethod
    def get_due_items(cls, limit: int = 20) -> list:
        """오늘 복습해야 할 항목들 반환"""
        data = cls._load_data()
        now = datetime.now()
        due_items = []

        for item_id, item in data["items"].items():
            try:
                next_review = datetime.fromisoformat(item["next_review"])
                if next_review <= now:
                    due_items.append(item)
            except:
                continue

        # 우선순위: 레벨 낮은 것 > 오래된 것
        due_items.sort(key=lambda x: (x["level"], x.get("next_review", "")))
        return due_items[:limit]

    @classmethod
    def get_stats(cls) -> dict:
        """학습 통계 반환"""
        data = cls._load_data()
        items = list(data["items"].values())

        if not items:
            return {
                "total_items": 0,
                "due_today": 0,
                "mastered": 0,
                "learning": 0,
                "new": 0,
                "total_reviews": data["stats"].get("total_reviews", 0),
                "avg_accuracy": 0,
            }

        now = datetime.now()
        due_today = 0
        mastered = 0
        new_items = 0

        for item in items:
            try:
                if datetime.fromisoformat(item["next_review"]) <= now:
                    due_today += 1
            except:
                pass

            if item["level"] >= 5:
                mastered += 1
            elif item["review_count"] == 0:
                new_items += 1

        total_correct = sum(i["correct_count"] for i in items)
        total_reviews = sum(i["review_count"] for i in items)

        return {
            "total_items": len(items),
            "due_today": due_today,
            "mastered": mastered,
            "learning": len(items) - mastered - new_items,
            "new": new_items,
            "total_reviews": total_reviews,
            "avg_accuracy": int((total_correct / total_reviews) * 100)
            if total_reviews > 0
            else 0,
        }

    @classmethod
    def get_category_stats(cls) -> dict:
        """카테고리별 통계"""
        data = cls._load_data()
        items = list(data["items"].values())

        category_stats = {}
        for item in items:
            cat = item.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {
                    "total": 0,
                    "mastered": 0,
                    "reviews": 0,
                    "correct": 0,
                }

            category_stats[cat]["total"] += 1
            category_stats[cat]["reviews"] += item["review_count"]
            category_stats[cat]["correct"] += item["correct_count"]
            if item["level"] >= 5:
                category_stats[cat]["mastered"] += 1

        return category_stats


# =====================================================
# 3-6. 학습 기록 관리 및 대시보드
# =====================================================


class LearningHistoryManager:
    """학습 기록 관리"""

    @staticmethod
    def _load_history() -> list:
        """학습 기록 로드"""
        if os.path.exists(LEARNING_HISTORY_FILE):
            try:
                with open(LEARNING_HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return []

    @staticmethod
    def _save_history(history: list):
        """학습 기록 저장"""
        with open(LEARNING_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    @classmethod
    def add_session(cls, session_data: dict):
        """학습 세션 기록 추가"""
        history = cls._load_history()
        session_data["timestamp"] = datetime.now().isoformat()
        session_data["date"] = datetime.now().strftime("%Y-%m-%d")
        history.append(session_data)
        cls._save_history(history)

        # 틀린 문제를 SRS에 추가
        wrong_items = session_data.get("wrong_items", [])
        for item in wrong_items:
            item_id = f"q_{item.get('question_id', item.get('id'))}_{datetime.now().strftime('%Y%m%d%H%M')}"
            SpacedRepetitionSystem.add_item(
                item_id=item_id,
                category=item.get("category", "comprehension"),
                content={
                    "question": item.get("question", ""),
                    "correct_answer": item.get("correct_answer", ""),
                    "evidence_quote": item.get("evidence_quote", ""),
                    "choices": item.get("choices", {}),
                },
            )

    @classmethod
    def get_recent_sessions(cls, limit: int = 10) -> list:
        """최근 학습 세션 목록"""
        history = cls._load_history()
        return history[-limit:][::-1]  # 최신순

    @classmethod
    def get_stats(cls) -> dict:
        """전체 학습 통계"""
        history = cls._load_history()

        if not history:
            return {
                "total_sessions": 0,
                "total_questions": 0,
                "total_correct": 0,
                "avg_score": 0,
                "sessions_this_week": 0,
                "score_trend": 0,
                "streak_days": 0,
            }

        # 기본 통계
        total_questions = sum(s.get("score", {}).get("total", 0) for s in history)
        total_correct = sum(s.get("score", {}).get("correct", 0) for s in history)
        scores = [s.get("score", {}).get("percent", 0) for s in history]
        avg_score = int(sum(scores) / len(scores)) if scores else 0

        # 이번 주 세션
        week_ago = datetime.now() - timedelta(days=7)
        sessions_this_week = sum(
            1
            for s in history
            if datetime.fromisoformat(s.get("timestamp", "2000-01-01")) > week_ago
        )

        # 점수 추세
        if len(history) >= 5:
            recent_avg = (
                sum(s.get("score", {}).get("percent", 0) for s in history[-5:]) / 5
            )
            older_avg = (
                sum(s.get("score", {}).get("percent", 0) for s in history[-10:-5]) / 5
                if len(history) >= 10
                else avg_score
            )
            score_trend = int(recent_avg - older_avg)
        else:
            score_trend = 0

        # 연속 학습일 계산
        streak_days = cls._calculate_streak(history)

        return {
            "total_sessions": len(history),
            "total_questions": total_questions,
            "total_correct": total_correct,
            "avg_score": avg_score,
            "sessions_this_week": sessions_this_week,
            "score_trend": score_trend,
            "streak_days": streak_days,
        }

    @classmethod
    def _calculate_streak(cls, history: list) -> int:
        """연속 학습일 계산"""
        if not history:
            return 0

        dates = set()
        for s in history:
            try:
                date = datetime.fromisoformat(s.get("timestamp", "")).date()
                dates.add(date)
            except:
                pass

        if not dates:
            return 0

        today = datetime.now().date()
        streak = 0
        current_date = today

        while current_date in dates or (
            current_date == today and (today - timedelta(days=1)) in dates
        ):
            if current_date in dates:
                streak += 1
            current_date -= timedelta(days=1)
            if current_date not in dates and current_date != today:
                break

        return streak

    @classmethod
    def get_weakness_analysis(cls, limit: int = 10) -> dict:
        """최근 세션들의 취약점 분석"""
        history = cls._load_history()
        recent = history[-limit:] if len(history) > limit else history

        all_wrong = []
        for session in recent:
            all_wrong.extend(session.get("wrong_items", []))

        return WeaknessAnalyzer.get_weakness_summary(all_wrong)

    @classmethod
    def get_daily_stats(cls, days: int = 7) -> list:
        """일별 학습 통계"""
        history = cls._load_history()
        daily = {}

        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            daily[date] = {"sessions": 0, "questions": 0, "correct": 0, "score_sum": 0}

        for session in history:
            date = session.get("date", "")
            if date in daily:
                daily[date]["sessions"] += 1
                daily[date]["questions"] += session.get("score", {}).get("total", 0)
                daily[date]["correct"] += session.get("score", {}).get("correct", 0)
                daily[date]["score_sum"] += session.get("score", {}).get("percent", 0)

        result = []
        for date in sorted(daily.keys()):
            d = daily[date]
            result.append(
                {
                    "date": date,
                    "sessions": d["sessions"],
                    "questions": d["questions"],
                    "correct": d["correct"],
                    "avg_score": int(d["score_sum"] / d["sessions"])
                    if d["sessions"] > 0
                    else 0,
                }
            )

        return result


# =====================================================
# 3-7. 유사 문제 생성
# =====================================================

SIMILAR_QUESTION_PROMPT = """You are an Indonesian language education expert.

Create ONE similar but different question based on the original question below.

Original Question:
- Id: {id}
- Question: {question}
- Choices: {choices_json}
- Correct Answer: {correct_answer}
- Evidence Quote: {evidence_quote}
- Category: {category}

CRITICAL REQUIREMENTS:
1. Keep the same category ({category}) and difficulty level
2. Test the same grammar/vocabulary concept but use different sentences/situations
3. Question and choices MUST be Bahasa Indonesia ONLY (no Hangul, no Korean words)
4. Korean is allowed ONLY in why_correct_ko (required, 1-2 sentences, non-empty)
5. evidence_quote MUST be Bahasa Indonesia ONLY (no Hangul)
6. The Indonesian sentence must be natural and grammatically correct
7. Create a completely new Indonesian sentence for evidence_quote that tests the same concept

RESPOND ONLY in this JSON format (no other text):
{
    "id": 99,
    "question": "Pertanyaan baru (Bahasa Indonesia only)",
    "category": "{category}",
    "choices": {
        "A": "Pilihan A (Bahasa Indonesia only)",
        "B": "Pilihan B (Bahasa Indonesia only)",
        "C": "Pilihan C (Bahasa Indonesia only)",
        "D": "Pilihan D (Bahasa Indonesia only)"
    },
    "answer": "A or B or C or D",
    "evidence_quote": "Kalimat baru bahasa Indonesia (Bahasa Indonesia only)",
    "why_correct_ko": "정답 해설 (Korean only, required, 1-2 sentences)"
}
"""

HANGUL_RE = re.compile(r"[가-힣]")


def _contains_hangul(text: str) -> bool:
    return bool(HANGUL_RE.search(text or ""))


def _has_hangul_in_question_or_choices(payload: dict) -> bool:
    question = payload.get("question", "")
    if _contains_hangul(question):
        return True
    choices = payload.get("choices", {})
    if isinstance(choices, dict):
        for choice in choices.values():
            if _contains_hangul(str(choice)):
                return True
    return False


def generate_similar_question(
    original_question: dict, model: str = "gpt-4o-mini"
) -> Optional[dict]:
    """원본 문제와 유사한 새 문제 생성"""

    category, _ = WeaknessAnalyzer.categorize_question(original_question)
    raw_choices = original_question.get("choices", {}) or {}
    clean_choices = {}
    if isinstance(raw_choices, dict):
        for key, value in raw_choices.items():
            clean_choices[key] = "" if _contains_hangul(str(value)) else value

    source = {
        "id": original_question.get("id"),
        "question": ""
        if _contains_hangul(original_question.get("question", ""))
        else original_question.get("question", ""),
        "choices": clean_choices,
        "answer": original_question.get("answer", ""),
        "evidence_quote": ""
        if _contains_hangul(original_question.get("evidence_quote", ""))
        else original_question.get("evidence_quote", ""),
    }

    prompt = safe_prompt_fill(
        SIMILAR_QUESTION_PROMPT,
        id=source.get("id", ""),
        question=source.get("question", ""),
        choices_json=json.dumps(source.get("choices", {}), ensure_ascii=True),
        category=category,
        correct_answer=source.get("answer", ""),
        evidence_quote=source.get("evidence_quote", ""),
    )

    try:
        result = llm_json(prompt, model=model)
        if _has_hangul_in_question_or_choices(result):
            strict_prompt = (
                prompt
                + "\n\nSTRICT OVERRIDE: The JSON must contain NO Hangul in question/choices. "
                "If any Hangul appears in question/choices, regenerate in Bahasa Indonesia only."
            )
            result = llm_json(strict_prompt, model=model)
            if _has_hangul_in_question_or_choices(result):
                return None
        result["is_similar"] = True
        result["original_id"] = original_question.get("id")
        result["category"] = category
        result["why_correct_ko"] = get_explanation_ko(result)
        return result
    except Exception as e:
        st.error(f"유사 문제 생성 실패: {e}")
        return None


def build_similar_repeat_seed_questions(mode: str, questions: list, model: str) -> list:
    seed_questions = []
    if not questions:
        return seed_questions

    used_ids = {str(q.get("id")) for q in questions if q.get("id") is not None}

    for idx, q in enumerate(questions, 1):
        similar = generate_similar_question(q, model=model)
        base = dict(similar) if similar else dict(q)
        original_id = str(q.get("id"))
        base["original_id"] = original_id
        base["is_similar"] = True

        suffix = 1
        candidate = f"{original_id}_s{suffix}"
        while candidate in used_ids:
            suffix += 1
            candidate = f"{original_id}_s{suffix}"
        base["id"] = candidate
        used_ids.add(candidate)
        seed_questions.append(base)

    return seed_questions


# =====================================================
# 3-8. TTS 섀도잉 기능 (edge-tts 안정화)
# =====================================================


async def generate_tts_audio(
    text: str,
    output_file: str,
    voice: str = "id-ID-ArdiNeural",
    rate: str = "+0%",
):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_file)


def _run_async(coro):
    """
    Streamlit/Windows 환경에서 이벤트루프 꼬임을 줄이기 위한 최소 안전 실행기.
    - 실행 중 루프가 없으면 asyncio.run
    - 실행 중 루프가 있으면 새 루프를 만들어 run_until_complete
    """
    try:
        running = asyncio.get_running_loop()
        # 여기에 오면 "이미 실행 중인 루프"가 있다는 뜻
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    except RuntimeError:
        # 실행 중 루프 없음
        return asyncio.run(coro)


def get_tts_audio_path(
    text: str,
    speed: str = "normal",
    voice: str = "id-ID-ArdiNeural",
) -> str | None:
    """
    TTS 오디오 파일 경로 반환 (캐시 사용)
    - 캐시 키에 voice + rate 포함 (중요!)
    - 파일이 0바이트/너무 작으면 재생성
    """
    if not text or not text.strip():
        return None

    # 캐시 폴더 보장
    os.makedirs(TTS_CACHE_DIR, exist_ok=True)

    # 속도에 따른 rate 설정
    speed_rates = {
        "very_slow": "-50%",
        "slow": "-25%",
        "normal": "+0%",
        "fast": "+25%",
    }
    rate = speed_rates.get(speed, "+0%")

    # ✅ 캐시 키에 voice + rate 포함 (이게 핵심)
    cache_key_src = f"v1|{voice}|{rate}|{speed}|{text}"
    cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
    audio_file = os.path.join(TTS_CACHE_DIR, f"{cache_key}.mp3")

    # 캐시된 파일이 있으면 (정상 크기인지도 체크)
    if os.path.exists(audio_file):
        try:
            if os.path.getsize(audio_file) > 1024:  # 1KB 이상이면 정상으로 간주
                return audio_file
        except OSError:
            pass
        # 너무 작거나 깨진 파일이면 삭제 후 재생성
        try:
            os.remove(audio_file)
        except OSError:
            pass

    # 생성
    try:
        _run_async(generate_tts_audio(text, audio_file, voice=voice, rate=rate))
        if os.path.exists(audio_file) and os.path.getsize(audio_file) > 1024:
            return audio_file
        return None
    except Exception as e:
        st.error(f"TTS 오디오 생성 실패: {e}")
        return None


def render_tts_player_edgetts(
    text: str,
    translation: str = "",
    speed: str = "normal",
    key_suffix: str = "",
    voice: str = "id-ID-ArdiNeural",
):
    """
    edge-tts를 사용한 TTS 재생 플레이어 렌더링
    - 실패 시 예외를 raise하지 않고 None 처리(상위에서 제어)
    """
    speed_label = TTS_SPEED_OPTIONS.get(speed, {}).get("label", speed)
    st.markdown(f"🇮🇩 {text}")
    if translation:
        st.markdown(f"🇰🇷 {translation}")
    st.caption(f"TTS Voice: {voice} / Speed: {speed_label}")

    with st.spinner("🎤 음성 생성 중..."):
        audio_file = get_tts_audio_path(text, speed=speed, voice=voice)

    if audio_file and os.path.exists(audio_file):
        st.audio(audio_file, format="audio/mp3")
        return True
    else:
        st.error("⚠️ 음성 생성에 실패했습니다.")
        return False


def render_tts_player(
    text: str, translation: str = "", speed: str = "normal", key_suffix: str = ""
):
    """
    TTS 재생 플레이어 렌더링 (edge-tts 우선)
    - 기본은 edge-tts만 사용(권장)
    - edge-tts 실패 시에만 fallback (원하면)
    """
    # ✅ edge-tts 우선
    ok = render_tts_player_edgetts(
        text, translation, speed, key_suffix, voice="id-ID-ArdiNeural"
    )
    if ok:
        return

    # (선택) fallback을 완전히 막고 싶으면 여기서 return 처리
    # st.warning("edge-tts 실패로 TTS를 중단합니다(한국어/기본 음성 폴백 방지).")
    # return

    # (선택) fallback: Web Speech API로 넘어가고 싶다면, 기존 WebSpeech 코드를 여기 두되,
    # 반드시 Indonesian voice 없으면 재생 금지하도록 유지하세요.
    st.warning(
        "⚠️ edge-tts 실패로 Web Speech API로 전환합니다. (환경에 따라 음성이 달라질 수 있음)"
    )
    # ... 기존 Web Speech API 블록 ...


# =====================================================
# 3-9. AI 학습 코치 UI 렌더링
# =====================================================


def render_ai_learning_coach(
    wrong_items: list, score_info: dict, condition: str, key_prefix: str = ""
):
    """
    AI 학습 코치 UI 렌더링

    Args:
        wrong_items: 틀린 문제 목록
        score_info: 점수 정보 (correct, total, percent)
        condition: 학습자 컨디션
        key_prefix: 키 접두사
    """
    st.markdown("#### 🤖 AI 학습 코치")

    if st.button(
        "💡 맞춤형 학습 조언 받기",
        type="secondary",
        width="stretch",
        key=f"{key_prefix}_ai_coach_btn",
    ):
        with st.spinner("AI 코치가 분석 중..."):
            # 취약 카테고리 분석
            categories = {}
            for item in wrong_items:
                cat = item.get("category", "기타")
                categories[cat] = categories.get(cat, 0) + 1

            weak_cats = ", ".join(
                [
                    f"{CEFR_CATEGORIES.get(k, {}).get('name', k)}({v}개)"
                    for k, v in categories.items()
                ]
            )

            # 틀린 문제 상세
            wrong_details_list = []
            for i, item in enumerate(wrong_items, 1):
                wrong_details_list.append(
                    f"{i}. {item.get('question', '')} "
                    f"(내 답: {item.get('user_answer')}, 정답: {item.get('correct_answer')})"
                )
            wrong_details = "\n".join(wrong_details_list)

            # AI 코치 프롬프트
            prompt = AI_LEARNING_COACH_PROMPT.format(
                score_percent=score_info.get("percent", 0),
                correct=score_info.get("correct", 0),
                total=score_info.get("total", 5),
                condition=condition if condition else "미설정",
                wrong_count=len(wrong_items),
                weak_categories=weak_cats if weak_cats else "없음",
                wrong_details=wrong_details,
            )

            try:
                ai_coach = llm_json(
                    prompt, model=st.session_state.get("gen_model", "gpt-4o-mini")
                )
                st.session_state[f"{key_prefix}_ai_coach"] = ai_coach
                st.success("✅ AI 코치 분석 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"AI 코치 생성 실패: {e}")

    # AI 코치 결과 표시
    if f"{key_prefix}_ai_coach" in st.session_state:
        ai_coach = st.session_state[f"{key_prefix}_ai_coach"]

        # 전반적인 평가
        st.markdown("##### 📊 전반적인 평가")
        st.info(ai_coach.get("overall_assessment", ""))

        # 강점과 약점
        col_strength, col_weakness = st.columns(2)
        with col_strength:
            st.markdown("**💪 강점**")
            for strength in ai_coach.get("strengths", []):
                st.markdown(f"- {strength}")
        with col_weakness:
            st.markdown("**🎯 개선 필요**")
            for weakness in ai_coach.get("weaknesses", []):
                st.markdown(f"- {weakness}")

        # 즉시 실행 액션
        st.markdown("##### ⚡ 지금 바로 할 일")
        for action in ai_coach.get("immediate_actions", []):
            with st.expander(
                f"🎯 {action.get('action', '')} ({action.get('time_needed', '')})"
            ):
                st.markdown(f"**이유:** {action.get('reason', '')}")

        # 주간 학습 계획
        st.markdown("##### 📅 1주일 학습 계획")
        for plan in ai_coach.get("weekly_plan", []):
            st.markdown(f"**{plan.get('day', '')}**: {plan.get('focus', '')}")
            for activity in plan.get("activities", []):
                st.markdown(f"  - {activity}")

        # 격려 메시지
        st.markdown("##### 💬 코치의 한마디")
        st.success(ai_coach.get("motivational_message", ""))

        # 추천 리소스
        if ai_coach.get("recommended_resources"):
            st.markdown("##### 📚 추천 학습 자료")
            for resource in ai_coach.get("recommended_resources", []):
                st.markdown(
                    f"- **[{resource.get('type', '')}] {resource.get('name', '')}**: {resource.get('description', '')}"
                )


# =====================================================
# 3-9. 반복 학습 UI 렌더링
# =====================================================


def render_repeat_learning_ui(mode: str, key_prefix: str = ""):
    """
    반복 학습 UI 렌더링 (인라인으로 사용 가능)

    Args:
        key_prefix: 키 접두사 (중복 방지)
    """
    progress = RepeatLearningManager.get_progress(mode)

    if not progress["active"]:
        return False  # 반복 학습이 활성화되지 않음

    # 진행 상황 표시
    st.markdown("### 📊 반복 학습 진행 중")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("완료", f"{progress['completed']}/{progress['total']}")
    with col2:
        st.metric("남은 문제", progress["remaining"])
    with col3:
        st.metric("총 시도", progress["total_retries"])

    st.progress(
        progress["progress_percent"] / 100,
        text=f"진행률: {progress['progress_percent']}%",
    )

    # 완료 체크
    if RepeatLearningManager.is_complete(mode):
        state = RepeatLearningManager.get_state(mode)
        if not state.get("celebration_shown", False):
            celebrate_confetti(
                key=f"repeat_complete_{mode}_{state.get('total_retries', 0)}"
            )
            state["celebration_shown"] = True
        mode_state = get_mode_state(mode)
        mode_state["retry_active"] = False
        mode_state["retry_completed"] = True
        mode_state["retry_mastered"] = True

        st.success("🎉 모든 문제를 정복했습니다! 훌륭해요!")

        render_shadowing_review_dropdown(
            navigate_to_page, mode, key_prefix=f"{key_prefix}_review"
        )

        # 완료 통계
        completed = state.get("completed", [])

        st.markdown("#### 📊 반복 학습 결과")
        for item in completed:
            retries = item.get("retries_needed", 1)
            emoji = "🌟" if retries == 1 else "✅" if retries <= 3 else "💪"
            st.markdown(f"{emoji} Q{item.get('id')}: {retries}번 만에 성공")

        col_restart, col_results, col_end = st.columns(3)
        with col_restart:
            if st.button(
                "🔄 처음부터 다시", key=f"{key_prefix}_repeat_restart", width="stretch"
            ):
                RepeatLearningManager.reset(mode)
                st.rerun()
        with col_results:
            if st.button(
                "📊 학습 결과 보기",
                type="primary",
                key=f"{key_prefix}_repeat_goto_results",
                width="stretch",
            ):
                navigate_to_page("results")
        with col_end:
            if st.button(
                "🏠 학습 종료", key=f"{key_prefix}_repeat_end", width="stretch"
            ):
                RepeatLearningManager.reset(mode)
                st.rerun()
        return True

    # 현재 문제 풀기
    current_q = RepeatLearningManager.get_next_question(mode)

    if not current_q:
        return False

    st.divider()

    # 문제 정보
    q_id = current_q.get("id", "?")
    is_similar = current_q.get("is_similar", False)

    if is_similar:
        st.markdown(f"### 🔄 유사 문제 (원본: Q{current_q.get('original_id', '?')})")
    else:
        st.markdown(f"### ❓ 문제 Q{q_id}")

    # 카테고리 표시
    category = current_q.get("category", "")
    if category:
        cat_info = CEFR_CATEGORIES.get(category, {"icon": "📌", "name": category})
        st.caption(f"{cat_info['icon']} {cat_info['name']}")

    # 문제
    question_text = current_q.get("question", "")
    if not question_text:
        st.error("⚠️ 문제를 불러올 수 없습니다. 반복 학습을 다시 시작해주세요.")
        if st.button("🔄 반복 학습 재시작", key=f"{key_prefix}_restart_error"):
            RepeatLearningManager.reset(mode)
            st.rerun()
        return False

    st.markdown(f"**{question_text}**")

    # 선택지
    choices = current_q.get("choices", {})

    with st.form(f"{key_prefix}_repeat_answer_form_{q_id}"):
        answer = st.radio(
            "답을 선택하세요",
            options=["A", "B", "C", "D"],
            format_func=lambda x: f"{x}. {choices.get(x, '')}",
            horizontal=True,
            index=None,
            key=f"{key_prefix}_repeat_answer_{q_id}",
        )

        col_submit, col_similar, col_stop = st.columns([2, 1, 1])
        with col_submit:
            submitted = st.form_submit_button(
                "✅ 제출", type="primary", width="stretch"
            )
        with col_similar:
            gen_similar = st.form_submit_button("🔄 유사 문제", width="stretch")
        with col_stop:
            stop_learning = st.form_submit_button("🛑 중단", width="stretch")

    # 중단 처리
    if stop_learning:
        RepeatLearningManager.reset(mode)
        st.info("반복 학습을 중단했습니다.")
        st.rerun()

    # 답안 제출 처리
    if submitted:
        if not answer:
            st.error("답을 선택해주세요!")
        else:
            is_correct, result = RepeatLearningManager.check_answer(mode, answer)
            update_shadowing_review_bank_answer(q_id, answer)

            if is_correct:
                st.success(f"🎉 정답입니다! ({result['retry_count']}번 만에 성공)")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ 오답입니다. 정답: {result['correct_answer']}")

                # 해설 표시
                why_correct = get_explanation_ko(current_q)
                if why_correct:
                    st.info(f"💡 **해설:** {why_correct}")
                else:
                    st.caption("해설이 제공되지 않았습니다")

                evidence = current_q.get("evidence_quote", "")
                if evidence:
                    st.markdown(f"📄 **근거:** _{evidence}_")

    # 유사 문제 생성
    if gen_similar:
        with st.spinner("유사 문제 생성 중..."):
            model_name = st.session_state.get("gen_model", "gpt-4o-mini")
            similar = generate_similar_question(current_q, model=model_name)
            if similar is None:
                similar = dict(current_q)
                similar["is_similar"] = True
                similar["original_id"] = current_q.get(
                    "original_id", current_q.get("id")
                )
            if similar:
                RepeatLearningManager.replace_with_similar(mode, similar)
                st.success("✅ 유사 문제가 생성되었습니다!")
                st.rerun()

    st.divider()
    render_shadowing_review_dropdown(
        navigate_to_page, mode, key_prefix=f"{key_prefix}_review_active"
    )

    st.markdown("#### 🗣️ 현재 문제 섀도잉")
    shadowing_answer = st.session_state.get(f"{key_prefix}_repeat_answer_{q_id}", "")
    shadowing_payload = build_shadowing_payload(
        mode, current_q, current_q, shadowing_answer or ""
    )
    if st.button(
        "🗣️ 섀도잉 연습하기",
        width="stretch",
        key=f"{key_prefix}_repeat_shadowing_{q_id}_{progress['total_retries']}",
    ):
        st.session_state["shadowing_return_to"] = "results"
        st.session_state["repeat_learning_resume"] = {
            "mode": mode,
            "q_id": str(q_id),
            "total_retries": progress.get("total_retries", 0),
        }
        st.session_state["force_review_tab"] = True
        set_shadowing_payload_and_go(shadowing_payload, navigate_to_page)

    return True


# =====================================================
# 3-9. 섀도잉 연습 섹션 렌더링
# =====================================================


def render_shadowing_section(coach_result: dict, speed: str = "normal"):
    """
    섀도잉 연습 섹션 렌더링

    Args:
        coach_result: 코칭 결과 (wrong_items, shadowing_sentences 포함)
        speed: TTS 속도
    """
    st.markdown("### 🗣️ 섀도잉 연습")
    st.info("💡 문장을 듣고 따라 말해보세요. 속도를 조절하여 연습할 수 있습니다.")

    # 속도 선택
    col1, col2 = st.columns([1, 3])
    with col1:
        speed = st.selectbox(
            "재생 속도",
            options=list(TTS_SPEED_OPTIONS.keys()),
            format_func=lambda x: TTS_SPEED_OPTIONS[x]["label"],
            index=2,  # normal
            key="shadowing_speed_select",
        )

    # 틀린 문제 근거 문장 (우선 표시)
    wrong_items = coach_result.get("wrong_items", [])
    evidence_quotes = [
        item.get("evidence_quote", "")
        for item in wrong_items
        if item.get("evidence_quote")
    ]

    if evidence_quotes:
        st.markdown("#### 📌 틀린 문제 근거 문장")
        st.caption("오답과 관련된 원문을 집중적으로 연습하세요.")

        for i, quote in enumerate(evidence_quotes):
            with st.expander(f"🔴 오답 근거 {i + 1}", expanded=(i == 0)):
                why_correct = (
                    wrong_items[i].get("why_correct_ko", "")
                    if i < len(wrong_items)
                    else ""
                )

                # ===== 30초 진단(디버그) =====
                st.write("DEBUG quote:", (quote or "")[:200])
                st.write(
                    "DEBUG contains_korean:",
                    any("가" <= ch <= "힣" for ch in (quote or "")),
                )
                # ============================

                render_tts_player(
                    text=quote,
                    translation=why_correct[:100] + "..."
                    if len(why_correct) > 100
                    else why_correct,
                    speed=speed,
                    key_suffix=f"evidence_{i}",
                )

    # 일반 섀도잉 문장
    shadowing_sentences = coach_result.get("shadowing_sentences", [])
    if shadowing_sentences:
        st.markdown("#### 📝 추가 연습 문장")

        for i, sentence in enumerate(shadowing_sentences):
            with st.expander(f"연습 {i + 1}", expanded=False):
                text = (
                    sentence.get("id", "")
                    if isinstance(sentence, dict)
                    else str(sentence)
                )
                translation = (
                    sentence.get("ko", "") if isinstance(sentence, dict) else ""
                )
                render_tts_player(
                    text=text,
                    translation=translation,
                    speed=speed,
                    key_suffix=f"shadow_{i}",
                )


def render_mode_shadowing_items(mode: str, items: List[Dict], speed: str) -> None:
    if not items:
        st.info("아직 이 모드에서 섀도잉 문장이 없습니다.")
        return

    for i, item in enumerate(items, 1):
        sentence = (item.get("sentence") or item.get("evidence_quote") or "").strip()
        if not sentence:
            continue
        question_id = item.get("question_id")
        header = f"연습 {i}"
        if question_id:
            header = f"Q{question_id} · {header}"
        with st.expander(header, expanded=(i == 1)):
            if item.get("evidence_quote") and item.get("evidence_quote") != sentence:
                st.caption(f"근거: {item.get('evidence_quote')}")
            render_tts_player(
                text=sentence,
                translation=item.get("source_snippet", ""),
                speed=speed,
                key_suffix=f"{mode}_shadow_{i}",
            )


# =====================================================
# 4. 채점 및 코칭 기능
# =====================================================


def compute_grade(quiz: dict, user_answers: dict):
    """
    퀴즈 결과를 채점합니다.

    Args:
        quiz: 퀴즈 JSON
        user_answers: 사용자 답안 딕셔너리

    Returns:
        tuple: (정답 수, 전체 문항 수, 정답률(%), 오답 목록)
    """
    questions = quiz.get("questions", [])
    correct_ids = []
    wrong_items = []

    for q in questions:
        qid = str(q.get("id"))
        correct = (q.get("answer") or "").strip()
        user = (user_answers.get(qid) or "").strip()

        if not qid or not correct or not user:
            continue

        if user == correct:
            correct_ids.append(qid)
        else:
            wrong_items.append(
                {
                    "id": int(qid) if qid.isdigit() else qid,
                    "user_answer": user,
                    "correct_answer": correct,
                }
            )

    total = len(questions) if questions else 5
    correct_n = len(correct_ids)
    percent = int(round((correct_n / total) * 100)) if total else 0

    return correct_n, total, percent, wrong_items


def sanitize_coach_structured(coach: dict, quiz: dict, user_answers: dict):
    """
    Structured Outputs로 생성된 코칭 결과를 검증하고 점수를 계산합니다.

    Args:
        coach: 코칭 결과 JSON (items 배열 포함)
        quiz: 퀴즈 JSON
        user_answers: 사용자 답안

    Returns:
        dict: 검증 및 점수가 추가된 코칭 결과
    """
    # None 체크
    if coach is None:
        coach = {
            "items": [],
            "weak_points_ko": [],
            "tomorrow_plan_10min_ko": [],
            "shadowing_sentences": [],
        }

    # 점수 계산
    correct_n, total, percent, _ = compute_grade(quiz, user_answers)

    # 점수 추가
    coach["score"] = {"correct": correct_n, "total": total, "percent": percent}

    # quiz의 questions를 딕셔너리로 변환 (evidence_quote 가져오기 위해)
    quiz_questions = quiz.get("questions", [])
    quiz_dict = {}
    for q in quiz_questions:
        qid = str(q.get("id"))
        quiz_dict[qid] = q

    # items의 각 항목 검증 및 보완
    items = coach.get("items", [])
    fixed_items = []

    for item in items:
        qid = str(item.get("id"))

        # quiz에서 evidence_quote 가져오기 (LLM이 복사하지 못한 경우 대비)
        evidence_from_quiz = ""
        if qid in quiz_dict:
            evidence_from_quiz = quiz_dict[qid].get("evidence_quote", "")

        # evidence_quote가 없으면 quiz에서 가져오기
        if not item.get("evidence_quote"):
            item["evidence_quote"] = evidence_from_quiz

        # summary_ko 보완
        if not item.get("summary_ko"):
            correct_explain = item.get("correct_explain_ko", "")
            correct_ans = quiz_dict.get(qid, {}).get("answer", "")
            evidence = item.get("evidence_quote", "")
            item["summary_ko"] = _build_item_summary(
                correct_explain, evidence, correct_ans
            )

        # choice_notes_ko 검증 (Pydantic이 보장하므로 항상 존재해야 함)
        choice_notes = item.get("choice_notes_ko", {})
        if not isinstance(choice_notes, dict):
            choice_notes = {}

        # 각 키가 없으면 기본값 설정 (Pydantic이 보장하지만 안전장치)
        if not all(k in choice_notes for k in ["A", "B", "C", "D"]):
            item["choice_notes_ko"] = {
                "A": choice_notes.get("A", "해설 없음"),
                "B": choice_notes.get("B", "해설 없음"),
                "C": choice_notes.get("C", "해설 없음"),
                "D": choice_notes.get("D", "해설 없음"),
            }

        fixed_items.append(item)

    coach["items"] = fixed_items

    # 하위 호환성을 위해 wrong_items도 생성
    wrong_items = []
    for item in fixed_items:
        if not item.get("is_correct", True):
            qid = str(item.get("id"))
            quiz_q = quiz_dict.get(qid, {})
            user_ans = user_answers.get(qid, "")
            correct_ans = quiz_q.get("answer", "")

            wrong_items.append(
                {
                    "id": item.get("id"),
                    "user_answer": user_ans,
                    "correct_answer": correct_ans,
                    "why_correct_ko": item.get("correct_explain_ko", ""),
                    "why_user_wrong_ko": item.get("wrong_reason_ko", ""),
                    "evidence_quote": item.get("evidence_quote", ""),
                    "choices_explanation": item.get("choice_notes_ko", {}),
                }
            )

    coach["wrong_items"] = wrong_items

    return coach


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    for sep in (". ", "? ", "! ", "。"):
        if sep in cleaned:
            return cleaned.split(sep)[0].strip() + sep.strip()
    return cleaned


def get_explanation_ko(q: dict) -> str:
    if not isinstance(q, dict):
        return ""
    for key in [
        "why_correct_ko",
        "explanation_ko",
        "why_correct",
        "explanation",
        "rationale_ko",
    ]:
        value = q.get(key, "")
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    evidence = (q.get("evidence_quote") or "").strip()
    answer = (q.get("answer") or "").strip()
    choices = q.get("choices") or {}
    choice_text = ""
    if answer and isinstance(choices, dict):
        choice_text = str(choices.get(answer, "")).strip()
    if evidence:
        if answer and choice_text:
            return f'근거 문장("{evidence}")의 의미상 정답은 {answer}({choice_text})입니다.'
        if answer:
            return f'근거 문장("{evidence}")의 의미상 정답은 {answer}입니다.'
        return f'근거 문장("{evidence}")을 근거로 정답을 판단하세요.'
    if answer and choice_text:
        return f"정답은 {answer}({choice_text})입니다."
    if answer:
        return f"정답은 {answer}입니다."
    return ""


def is_debug_mode() -> bool:
    if st.session_state.get("debug_mode_enabled"):
        return True

    try:
        qp = st.query_params
        debug_val = qp.get("debug")
        if isinstance(debug_val, list):
            debug_val = debug_val[0] if debug_val else None
        if str(debug_val).strip().lower() in ("1", "true", "yes", "y", "on"):
            st.session_state["debug_mode_enabled"] = True
            return True
    except Exception:
        pass

    return False


def _normalize_answer(a: str) -> str:
    return (a or "").strip().upper()


def _normalize_user_answers(m: dict) -> dict:
    if not isinstance(m, dict):
        return {}
    return {str(k): _normalize_answer(v) for k, v in m.items() if v is not None}


def compute_wrong_items(questions: list, user_answers: dict) -> tuple:
    wrong_items = []
    wrong_items_by_id = {}
    if not questions or not isinstance(user_answers, dict):
        return wrong_items, wrong_items_by_id

    for q in questions:
        q_id = str(q.get("id") or "").strip()
        if not q_id:
            continue
        correct_raw = q.get("answer", "")
        correct_norm = _normalize_answer(correct_raw)
        user_raw = user_answers.get(q_id, "")
        user_norm = _normalize_answer(user_raw)
        if not correct_norm:
            continue
        if user_norm != correct_norm:
            item = {
                "id": q_id,
                "question_id": q_id,
                "user_answer": user_raw,
                "correct_answer": correct_raw,
            }
            wrong_items.append(item)
            wrong_items_by_id[q_id] = item

    return wrong_items, wrong_items_by_id


def merge_wrong_items_for_repeat(
    coach_wrong_items: list, computed_wrong_items: list
) -> list:
    if not computed_wrong_items:
        return []
    coach_by_id = {str(item.get("id")): item for item in coach_wrong_items or []}
    merged = []
    for item in computed_wrong_items:
        q_id = str(item.get("id"))
        merged.append({**coach_by_id.get(q_id, {}), **item})
    return merged


def render_debug_line(
    *,
    qid: str,
    user_answer_raw: str,
    user_answer_norm: str,
    correct_answer_raw: str,
    correct_answer_norm: str,
    computed_is_correct,
    coach_is_correct,
    wrong_item_present: bool,
    evidence_present: bool,
) -> None:
    if not is_debug_mode():
        return

    st.caption(
        " · ".join(
            [
                f"`debug` qid={qid}",
                f"ua_raw={user_answer_raw!r}",
                f"ua={user_answer_norm or '∅'}",
                f"ca_raw={correct_answer_raw!r}",
                f"ca={correct_answer_norm or '∅'}",
                f"computed={computed_is_correct}",
                f"coach_is_correct={coach_is_correct}",
                f"wrong_item={wrong_item_present}",
                f"evidence={evidence_present}",
            ]
        )
    )


def _build_item_summary(
    correct_explain: str, evidence: str, correct_answer: str
) -> str:
    if correct_explain:
        return _first_sentence(correct_explain)
    if evidence and correct_answer:
        return f"정답은 {correct_answer}입니다. 근거 문장을 확인하세요."
    if correct_answer:
        return f"정답은 {correct_answer}입니다."
    return "정답 근거를 다시 확인해보세요."


def _normalize_choice_notes(choice_notes: dict) -> dict:
    if not isinstance(choice_notes, dict):
        choice_notes = {}
    return {
        "A": choice_notes.get("A", "지문/근거와 일치하지 않음"),
        "B": choice_notes.get("B", "지문/근거와 일치하지 않음"),
        "C": choice_notes.get("C", "지문/근거와 일치하지 않음"),
        "D": choice_notes.get("D", "지문/근거와 일치하지 않음"),
    }


def attach_explanations_to_quiz(quiz: dict, coach: dict, user_answers: dict) -> dict:
    if not quiz:
        return quiz

    items_by_id = {str(item.get("id")): item for item in coach.get("items", [])}
    for q in quiz.get("questions", []):
        qid = str(q.get("id"))
        item = items_by_id.get(qid, {})
        evidence = item.get("evidence_quote") or q.get("evidence_quote", "")
        correct_ans = q.get("answer", "")
        correct_explain = item.get("correct_explain_ko", "")
        summary = item.get("summary_ko") or _build_item_summary(
            correct_explain, evidence, correct_ans
        )
        choices_exp = _normalize_choice_notes(item.get("choice_notes_ko", {}))

        q["explanation"] = {
            "correct": correct_explain
            or _build_item_summary("", evidence, correct_ans),
            "options": choices_exp,
            "summary": summary,
        }
    return quiz


def pick_shadowing_target(question: dict, item: dict) -> str:
    evidence = (
        item.get("evidence_quote") or question.get("evidence_quote") or ""
    ).strip()
    if evidence:
        return evidence

    correct_key = question.get("answer", "")
    correct_choice = (question.get("choices", {}) or {}).get(correct_key, "")
    if correct_choice:
        return correct_choice

    snippet = (
        question.get("transcript_snippet")
        or question.get("source_text")
        or question.get("transcript")
        or question.get("question")
        or ""
    )
    return _first_sentence(snippet)


def build_shadowing_payload(
    mode: str, question: dict, item: dict, user_answer: str
) -> dict:
    source = "youtube" if mode == "video" else mode
    correct_answer = question.get("answer", "")
    rationale = (
        item.get("correct_explain_ko")
        or item.get("why_correct_ko")
        or item.get("summary_ko")
        or ""
    )
    evidence = item.get("evidence_quote") or question.get("evidence_quote")
    audio_ref = question.get("audio_ref") or question.get("audio_url") or None

    return {
        "id": str(question.get("id", item.get("id", ""))),
        "source": source,
        "target_text": pick_shadowing_target(question, item),
        "question": question.get("question", ""),
        "correct_answer": correct_answer,
        "user_answer": user_answer,
        "rationale": rationale,
        "evidence_quote": evidence,
        "audio_ref": audio_ref,
    }


def init_shadowing_review_bank(
    mode: str, wrong_items: list, quiz_questions: list
) -> list:
    q_dict = {str(q.get("id")): q for q in quiz_questions}
    bank = []
    for item in wrong_items:
        q_id = str(item.get("id") or item.get("question_id", ""))
        question = q_dict.get(q_id, {})
        target_question = question if question else item
        payload = build_shadowing_payload(
            mode, target_question, item, item.get("user_answer", "")
        )
        bank.append(
            {
                "id": str(payload.get("id") or q_id),
                "source": payload.get("source"),
                "target_text": payload.get("target_text"),
                "question": payload.get("question"),
                "correct_answer": payload.get("correct_answer"),
                "user_answer": payload.get("user_answer"),
                "rationale": payload.get("rationale"),
                "evidence_quote": payload.get("evidence_quote"),
                "audio_ref": payload.get("audio_ref"),
            }
        )
    st.session_state["shadowing_review_bank"] = bank
    if bank:
        selected_id = st.session_state.get("shadowing_review_selected")
        if selected_id not in {item.get("id") for item in bank}:
            st.session_state["shadowing_review_selected"] = bank[0].get("id")
    return bank


def update_shadowing_review_bank_answer(question_id: str, user_answer: str) -> None:
    bank = st.session_state.get("shadowing_review_bank", [])
    if not bank:
        return
    for item in bank:
        if str(item.get("id")) == str(question_id):
            item["user_answer"] = user_answer
            break


def _shadowing_review_label(item: dict) -> str:
    source_label = {
        "audio": "Audio",
        "youtube": "YouTube",
        "text": "Text",
    }.get(item.get("source"), "Source")
    target_text = (item.get("target_text") or "").strip().replace("\n", " ")
    snippet = (target_text[:30] + "...") if len(target_text) > 30 else target_text
    return f"[{source_label}] Q{item.get('id')} - {snippet}"


def render_shadowing_review_dropdown(
    navigate_to_page_fn, mode: str, key_prefix: str = "shadowing_review"
) -> None:
    bank = st.session_state.get("shadowing_review_bank", [])
    if not bank:
        return
    st.markdown("#### 🗂️ 섀도잉 복습 리스트")
    options = [item.get("id") for item in bank if item.get("id")]
    if not options:
        return
    selected_id = st.session_state.get("shadowing_review_selected")
    if selected_id not in options:
        selected_id = options[0]
    selected_id = st.selectbox(
        "복습할 섀도잉 항목 선택",
        options=options,
        index=options.index(selected_id),
        key=f"{key_prefix}_select",
        format_func=lambda opt: _shadowing_review_label(
            next((i for i in bank if i.get("id") == opt), {})
        ),
    )
    st.session_state["shadowing_review_selected"] = selected_id
    selected_item = next((item for item in bank if item.get("id") == selected_id), None)
    if st.button("🗣️ 섀도잉 연습하기", width="stretch", key=f"{key_prefix}_go"):
        if selected_item:
            st.session_state["shadowing_return_to"] = "results"
            state = RepeatLearningManager.get_state(mode)
            st.session_state["repeat_learning_resume"] = {
                "mode": mode,
                "q_id": str(selected_item.get("id")),
                "total_retries": state.get("total_retries", 0),
            }
            st.session_state["force_review_tab"] = True
            set_shadowing_payload_and_go(selected_item, navigate_to_page_fn)


def build_repeat_entry_shadowing_payload(
    mode: str, wrong_items: list, quiz_questions: list
):
    if not wrong_items:
        return None, ""
    first_item = wrong_items[0]
    q_id = str(first_item.get("id", ""))
    question = next((q for q in quiz_questions if str(q.get("id")) == q_id), {})
    target_question = question if question else first_item
    payload = build_shadowing_payload(
        mode,
        target_question,
        first_item,
        first_item.get("user_answer", ""),
    )
    return payload, str(target_question.get("id", first_item.get("id", "")))


def set_shadowing_payload_and_go(payload: dict, navigate_to_page_fn) -> None:
    st.session_state["shadowing_payload"] = payload
    navigate_to_page_fn("speaking")
    st.rerun()


def render_quiz_takeaways(coach: dict, wrong_items: list) -> None:
    items = coach.get("items", [])
    correct_notes = [
        i.get("correct_explain_ko", "") for i in items if i.get("correct_explain_ko")
    ]
    wrong_notes = [
        i.get("wrong_reason_ko", "")
        for i in items
        if not i.get("is_correct", True) and i.get("wrong_reason_ko")
    ]
    core_point = (
        _first_sentence(correct_notes[0])
        if correct_notes
        else "정답 근거를 원문에서 다시 확인해보세요."
    )
    confusion_point = (
        _first_sentence(wrong_notes[0])
        if wrong_notes
        else "이번 회차에서는 큰 혼동 포인트가 없었습니다."
    )
    if wrong_items:
        next_action = f"오답 {len(wrong_items)}문항은 섀도잉으로 복습해보세요."
    else:
        next_action = "정답률이 높습니다. 다음에는 난이도를 한 단계 올려보세요."

    st.markdown("#### 🧾 이번 퀴즈 결론/요약")
    st.markdown(f"- **핵심 포인트:** {core_point}")
    st.markdown(f"- **자주 헷갈린 포인트:** {confusion_point}")
    st.markdown(f"- **다음 행동:** {next_action}")


def render_question_result(
    mode: str,
    question: dict,
    item: dict,
    wrong_item: dict,
    user_answer: str,
    navigate_to_page_fn,
    key_prefix: str,
    wrong_items_by_id: Optional[dict] = None,
) -> None:
    qid = str(question.get("id"))
    question_text = question.get("question", "")
    item = item or {}
    correct_raw = (
        question.get("answer")
        or question.get("correct_answer")
        or item.get("correct_answer")
        or ""
    ).strip()
    correct_norm = _normalize_answer(correct_raw)
    ua_raw = (user_answer or "").strip()
    ua_norm = _normalize_answer(ua_raw)

    if ua_norm and correct_norm:
        computed_is_correct = ua_norm == correct_norm
    elif "is_correct" in item:
        computed_is_correct = bool(item.get("is_correct"))
    else:
        computed_is_correct = False

    coach_is_correct = item.get("is_correct") if item else None
    wrong_item_present = bool(wrong_item) or (
        qid in wrong_items_by_id if wrong_items_by_id is not None else False
    )
    evidence_present = bool(
        item.get("evidence_quote") or question.get("evidence_quote")
    )
    # computed_is_correct must be defined before status_emoji.
    status_emoji = "✅" if computed_is_correct else "❌"

    expander_label = f"Q{qid}. {question_text} {status_emoji}"
    with st.expander(expander_label, expanded=not computed_is_correct):
        correct_answer = correct_raw
        choices = question.get("choices", {}) or {}
        explanation = question.get("explanation", {})
        evidence = item.get("evidence_quote") or question.get("evidence_quote", "")

        st.markdown(f"**정답:** {correct_answer} | **내 답:** {ua_raw or '미선택'}")
        render_debug_line(
            qid=qid,
            user_answer_raw=ua_raw,
            user_answer_norm=ua_norm,
            correct_answer_raw=correct_raw,
            correct_answer_norm=correct_norm,
            computed_is_correct=computed_is_correct,
            coach_is_correct=coach_is_correct,
            wrong_item_present=wrong_item_present,
            evidence_present=evidence_present,
        )

        if computed_is_correct:
            correct_explain = explanation.get("correct") or item.get(
                "correct_explain_ko", ""
            )
            if correct_explain:
                st.markdown(f"✅ **왜 맞는가:** {correct_explain}")

            st.markdown("**선택지:**")
            for opt in ["A", "B", "C", "D"]:
                choice_text = choices.get(opt, "")
                label = f"{opt}. {choice_text}"
                if opt == correct_answer:
                    label = f"**{label}** ✅"
                st.markdown(f"- {label}")
        else:
            wrong_reason = item.get("wrong_reason_ko", "")
            if not wrong_reason and explanation.get("options"):
                wrong_reason = explanation.get("options", {}).get(ua_raw, "")
            if wrong_reason:
                st.markdown(f"❌ **내 답이 왜 틀렸는가:** {wrong_reason}")

            correct_explain = explanation.get("correct") or item.get(
                "correct_explain_ko", ""
            )
            if correct_explain:
                st.markdown(f"✅ **정답이 왜 맞는가:** {correct_explain}")

            options_exp = explanation.get("options", {})
            if options_exp:
                st.markdown("**📋 선택지별 설명:**")
                for opt in ["A", "B", "C", "D"]:
                    choice_text = choices.get(opt, "")
                    exp = options_exp.get(opt, "")
                    label = f"{opt}. {choice_text}"
                    if opt == correct_answer:
                        label = f"**{label}** ✅"
                    if opt == ua_raw and opt != correct_answer:
                        label = f"**{label}** ❌ 내 선택"
                    st.markdown(f"- {label} — {exp}")
        if evidence:
            st.caption(f'근거: "{evidence}"')

        summary = explanation.get("summary")
        if summary:
            st.caption(f"이 문항 요약: {summary}")

        if not computed_is_correct:
            st.caption("💡 반복 학습에서 섀도잉 연습을 진행할 수 있습니다.")


# =====================================================
# 5. Streamlit UI
# =====================================================

st.set_page_config(page_title=APP_TITLE, layout="wide")

inject_global_css(Path("ui/styles.css"))


# =====================================================
# 학습 설정 및 상태 초기화
# =====================================================
if "debug_mode_enabled" not in st.session_state:
    st.session_state["debug_mode_enabled"] = False
if "gen_model" not in st.session_state:
    st.session_state["gen_model"] = "gpt-4o-mini"
if "today_condition" not in st.session_state:
    st.session_state["today_condition"] = None
if "today_condition_note" not in st.session_state:
    st.session_state["today_condition_note"] = ""
if "learning_mode" not in st.session_state:
    st.session_state["learning_mode"] = None

LEVEL_OPTIONS = ["A1 (입문)", "A2 (초급)", "B1 (중급)", "B2 (중상급)"]
DIFFICULTY_PROFILE = {
    "A1 (입문)": {
        "cefr": "A1",
        "desc_ko": "기본 인사, 자기소개",
        "target_sentence_len_words": [4, 8],
        "max_subclauses": 0,
        "allowed_connectors": ["dan", "atau", "tapi"],
        "vocab_band": "top_1000_common",
        "distractor_subtlety": 1,
        "explanation_depth": "very short and simple (1 short sentence)",
        "question_type_mix": {
            "literal": 0.70,
            "vocab": 0.20,
            "grammar": 0.10,
            "inference": 0.00,
        },
        "grammar_targets": [
            "pronouns (saya/kamu/dia)",
            "simple nominal sentences",
            "basic negation (tidak/bukan)",
            "simple questions (apa/siapa/di mana)",
        ],
        "ban_list": [
            "relative clauses yang",
            "passive voice di-",
            "complex modality",
            "abstract nouns",
        ],
    },
    "A2 (초급)": {
        "cefr": "A2",
        "desc_ko": "일상 대화, 기본 문법",
        "target_sentence_len_words": [7, 12],
        "max_subclauses": 1,
        "allowed_connectors": ["dan", "atau", "tapi", "karena", "lalu"],
        "vocab_band": "top_2000_common",
        "distractor_subtlety": 2,
        "explanation_depth": "short and simple (1-2 short sentences)",
        "question_type_mix": {
            "literal": 0.55,
            "vocab": 0.20,
            "grammar": 0.15,
            "inference": 0.10,
        },
        "grammar_targets": [
            "simple past markers (sudah/pernah)",
            "future markers (akan/mau)",
            "prepositions (di/ke/dari)",
            "basic affixes (ber-, meN-) in common verbs",
        ],
        "ban_list": ["dense academic style", "multi-step inference > 1 hop"],
    },
    "B1 (중급)": {
        "cefr": "B1",
        "desc_ko": "의견 표현, 복잡한 문법",
        "target_sentence_len_words": [11, 18],
        "max_subclauses": 2,
        "allowed_connectors": [
            "karena",
            "tetapi",
            "sehingga",
            "walaupun",
            "jika",
            "agar",
        ],
        "vocab_band": "mid_frequency_and_topic_vocab",
        "distractor_subtlety": 3,
        "explanation_depth": "moderate detail (1-2 sentences)",
        "question_type_mix": {
            "literal": 0.40,
            "vocab": 0.20,
            "grammar": 0.20,
            "inference": 0.20,
        },
        "grammar_targets": [
            "complex negation nuance",
            "relative clauses with yang (simple)",
            "reported speech (kata/bilang) basic",
            "modality (harus/bisa/sebaiknya)",
        ],
        "ban_list": ["highly technical jargon without context"],
    },
    "B2 (중상급)": {
        "cefr": "B2",
        "desc_ko": "추상적 주제, 고급 문법",
        "target_sentence_len_words": [16, 26],
        "max_subclauses": 3,
        "allowed_connectors": [
            "meskipun",
            "namun",
            "sebaliknya",
            "oleh karena itu",
            "selain itu",
        ],
        "vocab_band": "advanced_and_abstract_vocab",
        "distractor_subtlety": 4,
        "explanation_depth": "deeper reasoning with nuance (2 sentences)",
        "question_type_mix": {
            "literal": 0.25,
            "vocab": 0.20,
            "grammar": 0.20,
            "inference": 0.35,
        },
        "grammar_targets": [
            "passive voice (di-) and causative (meN-kan) when appropriate",
            "nominalization and abstract nouns",
            "contrastive discourse markers",
            "paraphrase and implication",
        ],
        "ban_list": [],
    },
}

DEFAULT_LEVEL = "A2 (초급)"
LEGACY_MODE_TO_LEVEL = {
    "BIPA (초급)": "A2 (초급)",
    "BIPA (중급)": "B1 (중급)",
}

CONDITION_TO_QUESTIONS = {"A": 10, "B": 5, "C": 3}
CONDITION_LABELS = {"A": "A (여유)", "B": "B (보통)", "C": "C (힘듦)"}
DEFAULT_NUM_QUESTIONS = 5
LEGACY_CONDITION_TO_CODE = {label: code for code, label in CONDITION_LABELS.items()}

if st.session_state.get("today_condition") in LEGACY_CONDITION_TO_CODE:
    st.session_state["today_condition"] = LEGACY_CONDITION_TO_CODE[
        st.session_state.get("today_condition")
    ]

if "learning_level" not in st.session_state:
    legacy_mode = st.session_state.get("learning_mode")
    if legacy_mode in LEGACY_MODE_TO_LEVEL:
        st.session_state["learning_level"] = LEGACY_MODE_TO_LEVEL[legacy_mode]
    elif legacy_mode in LEVEL_OPTIONS:
        st.session_state["learning_level"] = legacy_mode
    else:
        st.session_state["learning_level"] = DEFAULT_LEVEL
if st.session_state.get("learning_level") not in LEVEL_OPTIONS:
    st.session_state["learning_level"] = DEFAULT_LEVEL
st.session_state["learning_mode"] = st.session_state.get("learning_level")


def get_learning_settings_state() -> dict:
    condition_code = st.session_state.get("today_condition")
    learning_level = st.session_state.get("learning_level")
    effective_condition = condition_code or "B"
    num_questions = CONDITION_TO_QUESTIONS.get(
        effective_condition, DEFAULT_NUM_QUESTIONS
    )
    level = learning_level if learning_level in LEVEL_OPTIONS else DEFAULT_LEVEL
    st.session_state["learning_mode"] = level
    return {
        "condition": condition_code,
        "learning_level": learning_level,
        "num_questions": num_questions,
        "level": level,
        "condition_simple": effective_condition,
    }


SUBCLAUSE_MARKERS = [
    "yang",
    "ketika",
    "walaupun",
    "meskipun",
    "karena",
    "sehingga",
    "agar",
    "jika",
    "lalu",
    "tapi",
    "tetapi",
    "namun",
    "sementara",
    "sebaliknya",
    "oleh karena itu",
    "selain itu",
]
QUESTION_TYPE_ORDER = ["literal", "vocab", "grammar", "inference"]


def get_difficulty_profile(level: str) -> dict:
    return DIFFICULTY_PROFILE.get(level, DIFFICULTY_PROFILE[DEFAULT_LEVEL])


def compute_word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b[\w'-]+\b", text))


def detect_subclause_count(text: str) -> int:
    if not text:
        return 0
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    count = 0
    for marker in SUBCLAUSE_MARKERS:
        pattern = r"\b" + re.escape(marker) + r"\b"
        count += len(re.findall(pattern, lowered))
    return count


def build_question_type_plan(profile: dict, num_questions: int) -> dict:
    mix = profile.get("question_type_mix", {})
    raw = {k: (mix.get(k, 0) * num_questions) for k in QUESTION_TYPE_ORDER}
    plan = {k: int(raw[k]) for k in QUESTION_TYPE_ORDER}
    remaining = max(num_questions - sum(plan.values()), 0)
    remainders = sorted(
        QUESTION_TYPE_ORDER,
        key=lambda k: (-(raw[k] - plan[k]), QUESTION_TYPE_ORDER.index(k)),
    )
    for k in remainders:
        if remaining <= 0:
            break
        plan[k] += 1
        remaining -= 1
    return plan


def difficulty_profile_prompt(level: str, num_questions: int) -> str:
    profile = get_difficulty_profile(level)
    plan = build_question_type_plan(profile, num_questions)
    plan_str = ", ".join([f"{k}={plan[k]}" for k in QUESTION_TYPE_ORDER])
    return (
        "DIFFICULTY_PROFILE:\n"
        f"  level: {profile['cefr']}\n"
        f"  target_sentence_len_words: {profile['target_sentence_len_words'][0]}-{profile['target_sentence_len_words'][1]}\n"
        f"  max_subclauses: {profile['max_subclauses']}\n"
        f"  vocab_band: {profile['vocab_band']}\n"
        f"  distractor_subtlety: {profile['distractor_subtlety']}/5\n"
        f"  explanation_depth: {profile['explanation_depth']}\n"
        f"  question_type_plan_for_N={num_questions}: {plan_str}\n"
        f"  grammar_targets: {profile['grammar_targets']}\n"
        f"  ban_list: {profile['ban_list']}\n"
    )


def build_difficulty_profile_used(level: str, num_questions: int) -> dict:
    profile = get_difficulty_profile(level)
    plan = build_question_type_plan(profile, num_questions)
    return {
        "level_label": level,
        "cefr": profile["cefr"],
        "target_sentence_len_words": profile["target_sentence_len_words"],
        "max_subclauses": profile["max_subclauses"],
        "vocab_band": profile["vocab_band"],
        "distractor_subtlety": profile["distractor_subtlety"],
        "explanation_depth": profile["explanation_depth"],
        "question_type_plan_for_N": plan,
        "grammar_targets": profile["grammar_targets"],
        "ban_list": profile["ban_list"],
    }


def normalize_quiz_difficulty(quiz: dict, level: str, num_questions: int) -> dict:
    profile = get_difficulty_profile(level)
    plan = build_question_type_plan(profile, num_questions)
    issues = []
    if not isinstance(quiz, dict):
        return {"issues": ["quiz_not_dict"], "plan": plan}

    quiz["level"] = profile["cefr"]
    quiz.setdefault(
        "difficulty_profile_used", build_difficulty_profile_used(level, num_questions)
    )

    questions = quiz.get("questions", [])
    type_counts = {k: 0 for k in QUESTION_TYPE_ORDER}
    word_range = profile["target_sentence_len_words"]
    max_subclauses = profile["max_subclauses"]
    word_violations = 0
    subclause_violations = 0

    for q in questions:
        q_type = str(q.get("type", "")).strip().lower()
        if q_type not in QUESTION_TYPE_ORDER:
            q_type = "literal"
            issues.append("invalid_question_type")
        q["type"] = q_type
        type_counts[q_type] += 1

        question_text = q.get("question", "")
        if question_text:
            wc = compute_word_count(question_text)
            if wc < word_range[0] or wc > word_range[1]:
                word_violations += 1
            if detect_subclause_count(question_text) > max_subclauses:
                subclause_violations += 1

        choices = q.get("choices", {}) or {}
        for choice_text in choices.values():
            wc = compute_word_count(choice_text)
            if wc < word_range[0] or wc > word_range[1]:
                word_violations += 1
            if detect_subclause_count(choice_text) > max_subclauses:
                subclause_violations += 1

    for q_type, expected in plan.items():
        if abs(type_counts.get(q_type, 0) - expected) > 1:
            issues.append("type_mix_off")
            break

    if word_violations > 0:
        issues.append("sentence_length_off")
    if subclause_violations > 0:
        issues.append("subclause_count_off")

    return {
        "issues": issues,
        "plan": plan,
        "type_counts": type_counts,
        "word_violations": word_violations,
        "subclause_violations": subclause_violations,
    }


def build_quiz_prompt(transcript: str, num_questions: int, level: str) -> str:
    profile = get_difficulty_profile(level)
    return safe_prompt_fill(
        QUIZ_PROMPT,
        num_questions=str(num_questions),
        transcript=transcript,
        level=profile["cefr"],
        level_profile=difficulty_profile_prompt(level, num_questions),
    )


def render_difficulty_warnings(check: dict, level: str, num_questions: int):
    if not check.get("issues"):
        return
    profile = get_difficulty_profile(level)
    plan = build_question_type_plan(profile, num_questions)
    st.warning(
        "난이도 체크 경고: "
        f"issues={check.get('issues')}, "
        f"type_counts={check.get('type_counts')}, "
        f"expected_plan={plan}, "
        f"word_violations={check.get('word_violations')}, "
        f"subclause_violations={check.get('subclause_violations')}"
    )


def generate_quiz_with_checks(
    transcript: str, num_questions: int, level: str, model: str, debug: bool
):
    prompt = build_quiz_prompt(transcript, num_questions, level)
    quiz = llm_json(prompt, model=model)
    check = normalize_quiz_difficulty(quiz, level, num_questions)
    if check.get("issues"):
        retry_prompt = (
            prompt + "\n\nSTRICT_RETRY: You MUST follow DIFFICULTY_PROFILE exactly. "
            "Enforce sentence length and max_subclauses for each question and choice, "
            "and match the question_type_plan_for_N."
        )
        quiz = llm_json(retry_prompt, model=model)
        check = normalize_quiz_difficulty(quiz, level, num_questions)
        if debug:
            render_difficulty_warnings(check, level, num_questions)
    elif debug:
        render_difficulty_warnings(check, level, num_questions)
    return quiz, prompt


def render_learning_top_controls(mode: str, navigate_to_page_fn, key_prefix: str):
    settings = get_learning_settings_state()
    condition = settings["condition"]
    learning_level = settings["learning_level"]
    num_questions = settings["num_questions"]
    level = settings["level"]

    def render_condition_controls():
        tmp_cond = f"{key_prefix}_tmp_today_condition"
        tmp_note = f"{key_prefix}_tmp_today_condition_note"
        if tmp_cond not in st.session_state:
            st.session_state[tmp_cond] = st.session_state.get("today_condition")
        if tmp_note not in st.session_state:
            st.session_state[tmp_note] = st.session_state.get(
                "today_condition_note", ""
            )
        st.selectbox(
            "오늘 컨디션",
            ["A", "B", "C"],
            index=None,
            placeholder="컨디션을 선택하세요",
            help="컨디션에 따라 문제 수가 달라집니다 (A: 10문제, B: 5문제, C: 3문제)",
            format_func=lambda k: CONDITION_LABELS.get(k, k),
            key=tmp_cond,
        )
        st.text_area("메모 (선택)", key=tmp_note, height=80)
        current_condition = st.session_state.get(tmp_cond)
        if current_condition:
            current_questions = CONDITION_TO_QUESTIONS.get(
                current_condition, DEFAULT_NUM_QUESTIONS
            )
            st.caption(f"💡 현재 설정: **{current_questions}문제** 생성")
        else:
            st.caption("⚠️ 컨디션을 선택하지 않았습니다 (기본: 5문제)")

    def render_condition_actions():
        tmp_cond = f"{key_prefix}_tmp_today_condition"
        tmp_note = f"{key_prefix}_tmp_today_condition_note"
        if st.button(
            "✅ 적용", type="primary", width="stretch", key=f"{key_prefix}_cond_apply"
        ):
            st.session_state["today_condition"] = st.session_state.get(tmp_cond)
            st.session_state["today_condition_note"] = st.session_state.get(
                tmp_note, ""
            )
            st.session_state.pop(tmp_cond, None)
            st.session_state.pop(tmp_note, None)
            st.rerun()

    def render_settings_controls():
        tmp_level = f"{key_prefix}_tmp_learning_level"
        if tmp_level not in st.session_state:
            st.session_state[tmp_level] = st.session_state.get("learning_level")
        st.selectbox(
            "학습 레벨",
            LEVEL_OPTIONS,
            index=None,
            placeholder="학습 레벨을 선택하세요",
            key=tmp_level,
        )
        current_level = st.session_state.get(tmp_level)
        if current_level:
            profile = get_difficulty_profile(current_level)
            st.caption(f"현재 레벨: {current_level} — {profile['desc_ko']}")
        else:
            profile = get_difficulty_profile(DEFAULT_LEVEL)
            st.caption(
                f"⚠️ 학습 레벨을 선택하지 않았습니다 (기본: {DEFAULT_LEVEL} — {profile['desc_ko']})"
            )

    def render_settings_actions():
        tmp_level = f"{key_prefix}_tmp_learning_level"
        if st.button(
            "✅ 적용", type="primary", width="stretch", key=f"{key_prefix}_mode_apply"
        ):
            st.session_state["learning_level"] = st.session_state.get(tmp_level)
            st.session_state["learning_mode"] = st.session_state.get("learning_level")
            st.session_state.pop(tmp_level, None)
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        condition_clicked = st.button(
            "🌤 오늘의 컨디션", width="stretch", key=f"{key_prefix}_condition_btn"
        )
    with col2:
        settings_clicked = st.button(
            "⚙️ 학습 설정", width="stretch", key=f"{key_prefix}_settings_btn"
        )

    if hasattr(st, "dialog"):

        @st.dialog("🌤 오늘의 컨디션")
        def condition_dialog():
            render_condition_controls()
            render_condition_actions()

        @st.dialog("⚙️ 학습 설정")
        def settings_dialog():
            render_settings_controls()
            render_settings_actions()

        if condition_clicked:
            condition_dialog()
        if settings_clicked:
            settings_dialog()
    else:
        show_condition = st.session_state.get(f"{key_prefix}_show_condition", False)
        show_settings = st.session_state.get(f"{key_prefix}_show_settings", False)
        if condition_clicked:
            show_condition = True
            st.session_state[f"{key_prefix}_show_condition"] = True
        if settings_clicked:
            show_settings = True
            st.session_state[f"{key_prefix}_show_settings"] = True
        with st.expander("🌤 오늘의 컨디션", expanded=show_condition):
            render_condition_controls()
            render_condition_actions()
        with st.expander("⚙️ 학습 설정", expanded=show_settings):
            render_settings_controls()
            render_settings_actions()

    if condition or learning_level:
        status_bits = []
        if condition:
            status_bits.append(
                f"오늘의 컨디션: {CONDITION_LABELS.get(condition, condition)}"
            )
        if learning_level:
            status_bits.append(f"설정: {learning_level} · {num_questions}문제")
        st.caption(" | ".join(status_bits))


# 디버그 모드 변수 (전역에서 사용 가능하도록)
debug = is_debug_mode()
gen_model = st.session_state.get("gen_model", "gpt-4o-mini")
_learning_settings = get_learning_settings_state()
condition = _learning_settings["condition"]
num_questions = _learning_settings["num_questions"]
level = _learning_settings["level"]

# =====================================================
# 페이지 네비게이션 시스템
# =====================================================

# 현재 페이지 초기화
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"


# 메인 홈으로 돌아가는 함수
def navigate_to_home():
    """메인 홈 화면으로 이동"""
    st.session_state["current_page"] = "home"
    st.rerun()


# 특정 페이지로 이동하는 함수
def navigate_to_page(page_name: str):
    """특정 페이지로 이동"""
    st.session_state["current_page"] = page_name
    st.rerun()


NAV_ITEMS = [
    ("audio", UI["nav_listening"]),
    ("youtube", UI["nav_watching"]),
    ("text", UI["nav_reading"]),
    ("results", UI["nav_progress"]),
    ("settings", UI["nav_settings"]),
]


def render_home_button_bottom(key: str = "home_bottom"):
    """페이지 하단 중앙에 홈 버튼 렌더링"""
    st.divider()
    left, center, right = st.columns([1, 2, 1])
    with center:
        if st.button(UI["btn_home"], key=key, width="stretch"):
            navigate_to_home()


def render_footer():
    """푸터 렌더링"""
    st.markdown(
        """
<div class="app-footer">
  <div class="footer-inner">
    <div class="section-label">Make it Bisa</div>
    <div style="font-size: 18px; margin: 12px 0 24px 0;">
      꾸준한 학습으로 당신만의 Bisa를 만들어보세요.
    </div>
    <div style="margin-bottom: 16px;">
      <a href="#">Privacy Policy</a>
      <a href="#">Terms of Use</a>
      <a href="#">Accessibility</a>
    </div>
    <div style="font-size: 14px; line-height: 1.6;">
      <strong>🔒 개인정보 보호 및 저작권 준수</strong>
      <div>- YouTube 영상은 임베드 형태로만 제공되며, 자동 다운로드하지 않습니다.</div>
      <div>- 웹 크롤링은 공개된 교육 자료에 한해 제공되며, 저작권을 준수합니다.</div>
      <div>- 생성된 퀴즈 및 코칭 내용은 원본 텍스트를 1:1 복사하지 않고 재작성됩니다.</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# 메인 홈 화면
# =====================================================


def _img_data_uri(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    else:
        # 알 수 없는 확장자는 png로 가정
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def render_home_page():
    hero_path = Path("assets/hero.png")

    if hero_path.exists():
        hero_img = f'<img class="hero-media-img" src="{_img_data_uri(hero_path)}" alt="Learner listening on a laptop" />'
    else:
        # 파일이 없을 때 기존 프레임(그라데이션) fallback
        hero_img = '<div class="hero-media-frame"></div>'

    st.markdown(
        f"""
<div class="home-hero">
  <div class="hero-content">
    <div class="hero-title">{UI["home_title"]}</div>
    <div class="hero-subtext">{UI["home_subtext"]}</div>
    <a class="hero-cta" href="#card-grid">{UI["home_cta_prompt"]}</a>
  </div>
  <div class="hero-media">
    {hero_img}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div id="card-grid"></div>', unsafe_allow_html=True)

    # Duplicate home nav cards removed to avoid double-rendering under the hero.


# =====================================================
# 페이지 함수들
# =====================================================


def render_audio_page():
    """오디오 학습 페이지 렌더링"""
    gen_model = st.session_state.get("gen_model", "gpt-4o-mini")

    st.header("🎵 오디오로 학습하기")
    st.markdown("WAV 파일을 업로드하면 음성을 텍스트로 변환하고 퀴즈를 생성합니다.")

    render_learning_top_controls("audio", navigate_to_page, key_prefix="audio_top")

    # 1단계: 오디오 선택
    st.subheader("1️⃣ 오디오 선택")

    col_audio1, col_audio2 = st.columns([3, 1])

    with col_audio1:
        use_sample = st.checkbox("샘플 오디오 사용", value=True, key="use_sample_audio")

    # uploaded 변수 초기화 (변수 정의 오류 방지)
    uploaded = None

    with col_audio2:
        if not use_sample:
            uploaded = st.file_uploader(
                "WAV 업로드",
                type=["wav"],
                key="audio_uploader",
                label_visibility="collapsed",
            )

    wav_path = None
    if use_sample:
        # 샘플 파일 경로 수정 (프로젝트 루트에 위치)
        sample_paths = [
            f for f in ["sample_A.wav", "sample_B.wav"] if os.path.exists(f)
        ]
        if sample_paths:
            sample_choice = st.radio(
                "샘플 선택", sample_paths, index=0, horizontal=True
            )
            wav_path = sample_choice
        else:
            # 샘플 파일이 없을 경우 안내 메시지
            st.warning(
                "⚠️ 샘플 오디오 파일(sample_A.wav, sample_B.wav)을 찾을 수 없습니다. 파일을 업로드해주세요."
            )
    else:
        if uploaded is not None:
            temp_path = os.path.join(LOG_DIR, f"upload_{int(time.time())}.wav")
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            wav_path = temp_path

    # 오디오 재생
    if wav_path:
        st.audio(wav_path, format="audio/wav")

        # ASR 실행 버튼
        if st.button(
            "🎤 음성 → 텍스트 변환", type="primary", key="btn_asr", width="stretch"
        ):
            asr_pipe = load_asr()  # @st.cache_resource라 1회 로드 후 재사용
            t0 = time.perf_counter()

            try:
                with st.spinner(
                    "음성을 텍스트로 변환 중... (CPU에서는 시간이 걸릴 수 있습니다)"
                ):
                    transcript = core_transcribe_audio(
                        asr_pipe,
                        wav_path,
                        target_sr=TARGET_SR,
                        language="indonesian",
                    )

                    # 가독성을 위해 오디오 전용 포맷팅 적용 (문장 단위로 3개씩 문단 구분)
                    formatted_transcript = format_audio_transcript(
                        transcript, sentences_per_paragraph=3
                    )
                    st.session_state["audio_transcript"] = formatted_transcript
                    st.session_state["current_source"] = (
                        f"Audio: {os.path.basename(wav_path)}"
                    )

                    # 퀴즈 초기화
                    st.session_state.pop("audio_quiz", None)
                    st.session_state.pop("audio_coach", None)
                    reset_mode_ephemeral("audio")
                    RepeatLearningManager.reset("audio")

                dt = time.perf_counter() - t0
                st.success(f"✅ 변환 완료! ({dt:.1f}초 소요)")
                st.rerun()

            except Exception as e:
                st.error("❌ 변환 실패")
                st.exception(e)
    else:
        st.info("👆 오디오 파일을 선택해주세요.")

    # 2단계: 변환된 텍스트
    audio_transcript = st.session_state.get("audio_transcript", "")

    if audio_transcript:
        st.divider()
        st.subheader("2️⃣ 변환된 텍스트 (문단별로 구분됨)")

        st.text_area(
            "인도네시아어 텍스트",
            value=audio_transcript,
            height=500,
            key="audio_transcript_display",
            disabled=True,
        )
        # 문단 수 계산
        paragraph_count = audio_transcript.count("\n\n") + 1
        st.caption(
            f"📊 텍스트 길이: {len(audio_transcript)}자 | 문단 수: {paragraph_count}개"
        )

        # 퀴즈 생성 버튼
        st.markdown("---")

        if st.button(
            f"🎯 퀴즈 {num_questions}문항 생성하기",
            type="primary",
            key="btn_generate_audio_quiz_main",
            width="stretch",
        ):
            st.session_state["start_audio_quiz_generation"] = True
            st.rerun()

    # 퀴즈 생성 처리
    if st.session_state.get("start_audio_quiz_generation"):
        st.divider()
        st.subheader("3️⃣ 퀴즈 생성 중...")

        if audio_transcript:
            try:
                quiz_text = (
                    audio_transcript[:4000]
                    if len(audio_transcript) > 4000
                    else audio_transcript
                )
                with st.spinner("퀴즈를 생성 중... (약 10초 소요)"):
                    base_profile = get_level_profile(
                        level
                    )  # level이 "A1"~"C2" 문자열이라 가정
                    profile = {
                        "level": base_profile.level,
                        "max_questions": num_questions,
                        "target_vocab_band": base_profile.target_vocab_band,
                        "num_questions": num_questions,
                        "cefr_level": level,
                        "level_profile": difficulty_profile_prompt(
                            level, num_questions
                        ),
                    }
                    profile["level"] = level
                    quiz_result = core_generate_quiz(
                        client=client,
                        model=gen_model,
                        template=QUIZ_PROMPT,
                        transcript=quiz_text,
                        profile=profile,
                    )

                    quiz = quiz_result.payload
                    prompt = None  # debug용 prompt가 필요하면 core/quiz.py에서 반환하도록 확장 가능

                if debug:
                    with st.expander("🔍 DEBUG: QUIZ_PROMPT"):
                        st.code(prompt[:1000])

                st.session_state["audio_quiz"] = quiz
                st.session_state.pop("audio_coach", None)
                st.session_state.pop("start_audio_quiz_generation")
                st.success("✅ 퀴즈 생성 완료!")
                st.rerun()

            except Exception as e:
                st.error("❌ 퀴즈 생성 실패")
                st.exception(e)
                st.session_state.pop("start_audio_quiz_generation", None)

    # 퀴즈 표시 및 답안 입력
    audio_quiz = st.session_state.get("audio_quiz")

    if audio_quiz:
        st.divider()
        st.subheader("3️⃣ 퀴즈 풀이")

        audio_quiz = st.session_state.get("audio_quiz")

        if audio_quiz:
            questions = audio_quiz.get("questions", [])

            if questions:
                with st.form("audio_quiz_form"):
                    user_answers = {}

                    for q in questions:
                        qid = q.get("id")
                        st.markdown(f"**Q{qid}. {q.get('question', '')}**")

                        choices = q.get("choices", {})
                        opts = ["A", "B", "C", "D"]

                        # 초기에 아무것도 선택되지 않도록 index=None 설정
                        pick = st.radio(
                            "답 선택",
                            options=opts,
                            format_func=lambda k,
                            choices=choices: f"{k}. {choices.get(k, '')}",
                            key=f"audio_q_{qid}",
                            index=None,
                            horizontal=True,
                        )
                        user_answers[str(qid)] = pick if pick else ""
                        st.divider()

                    submitted = st.form_submit_button("✅ 채점하기", type="primary")

                # 채점 및 코칭
                if submitted:
                    # 모든 답안이 선택되었는지 확인
                    empty_answers = [
                        qid for qid, ans in user_answers.items() if not ans
                    ]
                    if empty_answers:
                        st.error(
                            f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})"
                        )
                    else:
                        try:
                            condition_simple = condition or "B"

                            # 1) 순수 채점(점수/오답) 먼저 계산해서 저장
                            grading = grade_quiz(audio_quiz, user_answers)
                            st.session_state["audio_grading"] = {
                                "score": grading.score,
                                "total": grading.total,
                                "wrong_items": [
                                    wi.__dict__ for wi in grading.wrong_items
                                ],
                            }

                            st.session_state["audio_user_answers"] = (
                                _normalize_user_answers(user_answers)
                            )

                            prompt = safe_prompt_fill(
                                COACH_PROMPT,
                                transcript=(
                                    audio_transcript[:4000]
                                    if audio_transcript and len(audio_transcript) > 4000
                                    else (audio_transcript or "")
                                ),
                                quiz_json=json.dumps(audio_quiz, ensure_ascii=False),
                                user_answers_json=json.dumps(
                                    user_answers, ensure_ascii=False
                                ),
                                condition=condition_simple,
                            )

                            with st.spinner("채점 중..."):
                                # Structured Outputs 사용
                                coach = llm_structured(
                                    prompt, CoachResponse, model=gen_model
                                )
                                coach = sanitize_coach_structured(
                                    coach, audio_quiz, user_answers
                                )

                            audio_quiz = attach_explanations_to_quiz(
                                audio_quiz, coach, user_answers
                            )
                            st.session_state["audio_quiz"] = audio_quiz

                            st.session_state["audio_coach"] = coach

                            # 학습 기록 저장
                            wrong_items_analyzed = []
                            for item in coach.get("wrong_items", []):
                                q = next(
                                    (
                                        q
                                        for q in audio_quiz.get("questions", [])
                                        if str(q.get("id")) == str(item.get("id"))
                                    ),
                                    {},
                                )
                                analyzed = WeaknessAnalyzer.analyze_wrong_answer(
                                    q,
                                    item.get("user_answer", ""),
                                    item.get("correct_answer", ""),
                                )
                                analyzed.update(item)
                                # ID 필드 명시적으로 설정 (SRS에 추가되도록)
                                q_id = str(q.get("id", item.get("id", "")))
                                analyzed["id"] = q_id
                                analyzed["question_id"] = q_id
                                wrong_items_analyzed.append(analyzed)

                            LearningHistoryManager.add_session(
                                {
                                    "source": "audio",
                                    "level": level,
                                    "condition": condition,
                                    "score": coach.get("score", {}),
                                    "wrong_items": wrong_items_analyzed,
                                }
                            )

                            record_mode_result(
                                "audio",
                                {
                                    "source": "audio",
                                    "level": level,
                                    "condition": condition,
                                    "score": coach.get("score", {}),
                                    "wrong_items": wrong_items_analyzed,
                                    "quiz_questions": audio_quiz.get("questions", []),
                                    "coach": coach,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )

                            st.success("✅ 채점 완료!")
                            st.rerun()

                        except Exception as e:
                            st.error("❌ 채점 실패")
                            st.exception(e)

                            # 디버그 모드에서 상세 정보 표시
                            if debug:
                                if "last_llm_response" in st.session_state:
                                    with st.expander("🔍 DEBUG: 오류 상세 정보"):
                                        st.json(st.session_state["last_llm_response"])

        # 코칭 결과 표시
        audio_coach = st.session_state.get("audio_coach")

        if audio_coach:
            st.divider()
            st.markdown("### 🎓 학습 결과")

            # 점수 표시
            score = audio_coach.get("score", {})
            correct = score.get("correct", 0)
            total = score.get("total", 5)
            percent = score.get("percent", 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("정답 수", f"{correct}/{total}")
            col2.metric("정답률", f"{percent}%")
            col3.metric("컨디션", condition or "미설정")

            # 오답 풀이 및 해설
            st.divider()
            st.markdown("#### 📝 문제 풀이 및 해설")

            audio_quiz = st.session_state.get("audio_quiz", {})
            questions = audio_quiz.get("questions", [])
            wrong_items = audio_coach.get("wrong_items", [])
            wrong_items_by_id = {str(item.get("id")): item for item in wrong_items}
            items_by_id = {
                str(item.get("id")): item for item in audio_coach.get("items", [])
            }
            user_answers = st.session_state.get("audio_user_answers", {})
            computed_wrong_items, _ = compute_wrong_items(questions, user_answers)
            wrong_items_for_repeat = merge_wrong_items_for_repeat(
                wrong_items, computed_wrong_items
            )
            all_correct = len(computed_wrong_items) == 0

            for q in questions:
                qid = str(q.get("id"))
                item = items_by_id.get(qid, {})
                wrong_item = wrong_items_by_id.get(qid, {})
                user_ans = user_answers.get(qid) or wrong_item.get("user_answer", "")
                render_question_result(
                    mode="audio",
                    question=q,
                    item=item,
                    user_answer=user_ans,
                    wrong_item=wrong_item,
                    wrong_items_by_id=wrong_items_by_id,
                    key_prefix="audio",
                    navigate_to_page_fn=navigate_to_page,
                )

            st.divider()
            render_quiz_takeaways(audio_coach, wrong_items)

            # 취약 포인트
            st.divider()
            st.markdown("#### 🎯 취약 포인트")
            for wp in audio_coach.get("weak_points_ko", []):
                st.markdown(f"- {wp}")

            # 내일 학습 플랜
            st.markdown("#### 📅 내일 10분 학습 플랜")
            for step in audio_coach.get("tomorrow_plan_10min_ko", []):
                st.markdown(f"- **{step.get('minute', '')}분**: {step.get('task', '')}")

            # Shadowing 문장
            st.markdown("#### 🗣️ Shadowing 연습")
            for s in audio_coach.get("shadowing_sentences", []):
                st.markdown(f"**{s.get('id', '')}**")
                st.markdown(f"→ _{s.get('ko', '')}_")
                st.markdown("")

            # AI 학습 코치 (코치 결과 기반)
            if wrong_items and len(wrong_items) > 0:
                st.divider()
                render_ai_learning_coach(
                    wrong_items=wrong_items,
                    score_info={"correct": correct, "total": total, "percent": percent},
                    condition=condition,
                    key_prefix="audio",
                )

            st.divider()

            # 반복 학습이 이미 진행 중인지 확인
            repeat_progress = RepeatLearningManager.get_progress("audio")

            if is_debug_mode():
                st.caption(
                    f"[debug] mode=audio all_correct={all_correct} "
                    f"wrong_count={len(computed_wrong_items)} "
                    f"repeat_active={repeat_progress['active']}"
                )

            st.markdown("#### 🔄 반복 학습")

            if repeat_progress["active"]:
                render_repeat_learning_ui("audio", key_prefix="audio")
            else:
                if not all_correct:
                    st.info(
                        f"💡 틀린 문제 {len(computed_wrong_items)}개를 모두 맞출 때까지 반복 학습할 수 있습니다!"
                    )

                    if st.button(
                        "🚀 틀린 문제 반복 학습 시작",
                        type="primary",
                        width="stretch",
                        key="audio_start_repeat",
                    ):
                        # 취약점 분석 추가
                        analyzed_wrong = []
                        for item in wrong_items_for_repeat:
                            q_id = str(item.get("id"))
                            orig_q = next(
                                (q for q in questions if str(q.get("id")) == q_id), {}
                            )
                            analyzed = WeaknessAnalyzer.analyze_wrong_answer(
                                orig_q,
                                item.get("user_answer", ""),
                                item.get("correct_answer", ""),
                            )
                            analyzed["why_correct_ko"] = item.get("why_correct_ko", "")
                            analyzed["why_user_wrong_ko"] = item.get(
                                "why_user_wrong_ko", ""
                            )
                            analyzed_wrong.append(analyzed)

                        # 반복 학습 시작
                        RepeatLearningManager.start_repeat_learning(
                            "audio", analyzed_wrong, questions
                        )
                        st.rerun()
                    entry_payload, entry_q_id = build_repeat_entry_shadowing_payload(
                        "audio",
                        wrong_items_for_repeat,
                        questions,
                    )
                    if entry_payload and st.button(
                        "🗣️ 섀도잉 연습하기",
                        width="stretch",
                        key="audio_repeat_shadowing_entry",
                    ):
                        st.session_state["shadowing_return_to"] = "results"
                        st.session_state["repeat_learning_resume"] = {
                            "mode": "audio",
                            "q_id": entry_q_id,
                            "total_retries": 0,
                        }
                        st.session_state["force_review_tab"] = True
                        set_shadowing_payload_and_go(entry_payload, navigate_to_page)
                else:
                    st.success(
                        "🎉 모든 문제 정답입니다! 이제 유사 문제로 한 번 더 다져볼까요?"
                    )
                    if st.button(
                        "🔄 유사 문제로 반복 학습 시작",
                        type="primary",
                        width="stretch",
                        key="audio_start_similar_repeat",
                    ):
                        with st.spinner("유사 문제를 생성 중..."):
                            similar_questions = build_similar_repeat_seed_questions(
                                "audio",
                                questions,
                                model=gen_model,
                            )
                        RepeatLearningManager.start_custom(
                            "audio", similar_questions, kind="similar"
                        )
                        st.rerun()

            # 학습 결과 페이지로 이동 버튼
            st.divider()
            if st.button(
                UI["btn_view_progress"],
                type="primary",
                width="stretch",
                key="audio_goto_results",
            ):
                navigate_to_page("results")

    # ✅ 페이지 하단 중앙 홈 버튼
    render_home_button_bottom(key="home_from_audio_bottom")


def render_youtube_page():
    """YouTube 학습 페이지 렌더링"""
    gen_model = st.session_state.get("gen_model", "gpt-4o-mini")

    st.header("📺 YouTube로 학습하기")
    st.markdown(
        "YouTube 영상을 시청하고 인도네시아어 요약을 작성한 후 퀴즈를 풀어보세요!"
    )

    st.warning("""
    ⚠️ **중요 사항**:
    - YouTube 영상은 임베드 형태로만 제공됩니다.
    - 자동으로 자막이나 오디오를 다운로드하지 않습니다.
    - 사용자가 직접 시청하고 메모한 내용을 입력해주세요.
    """)

    render_learning_top_controls("youtube", navigate_to_page, key_prefix="youtube_top")

    # YouTube URL 입력
    st.subheader("1️⃣ YouTube 영상 선택")

    col_url1, col_url2 = st.columns([3, 1])

    # 샘플 로드 플래그 확인 (이전 rerun에서 설정된 경우)
    if st.session_state.get("load_sample_flag"):
        st.session_state["youtube_url_input"] = (
            "https://www.youtube.com/watch?v=_j3ixl3EH6M&t=4s"
        )
        st.session_state.pop("load_sample_flag")  # 플래그 제거
        reset_learning_state("youtube")
        st.session_state.pop("prev_youtube_video_id", None)

    with col_url1:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
            help="YouTube 영상 URL을 입력하세요",
        )

    with col_url2:
        st.markdown("**샘플 링크**")
        if st.button("📺 샘플 로드", key="load_sample"):
            # 플래그만 설정하고 rerun (다음 실행에서 URL 설정)
            st.session_state["load_sample_flag"] = True
            st.rerun()

    # URL 변경 감지 및 초기화 (video_id 기준)
    if youtube_url:
        current_video_id = extract_youtube_id(youtube_url)
        prev_video_id = st.session_state.get("prev_youtube_video_id", "")

        # 자막 가져오기 중인지 확인 (초기화 방지)
        fetching_transcript = st.session_state.get("fetching_transcript", False)

        if (
            prev_video_id
            and prev_video_id != current_video_id
            and not fetching_transcript
        ):
            # video_id가 변경되었으면 이전 데이터가 있는지 확인
            had_data = (
                st.session_state.get("youtube_quiz") is not None
                or st.session_state.get("youtube_coach") is not None
                or st.session_state.get("youtube_transcript") is not None
            )

            # 상태 초기화 함수 사용 (video_id가 다르므로 초기화됨)
            reset_learning_state("youtube", current_video_id)

            # 데이터가 있었을 경우에만 알림
            if had_data:
                st.info(f"🔄 새로운 영상(`{current_video_id}`)으로 변경되었습니다.")

        # 현재 video_id 기록
        if current_video_id:
            st.session_state["prev_youtube_video_id"] = current_video_id

        # fetching_transcript 플래그 제거
        st.session_state.pop("fetching_transcript", None)

    # YouTube 임베드
    if youtube_url:
        video_id = extract_youtube_id(youtube_url)

        if video_id:
            st.markdown(
                f"""
            <iframe width="100%" height="400" 
            src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
            </iframe>
            """,
                unsafe_allow_html=True,
            )

            st.caption(f"출처: YouTube | {youtube_url}")
            st.session_state["current_source"] = f"YouTube: {youtube_url}"

            # 현재 영상 정보 표시
            st.info(f"📹 **현재 영상 ID:** `{video_id}`")

            # 기존 퀴즈가 다른 영상의 것인지 확인
            saved_video_id = st.session_state.get("youtube_quiz_video_id", "")
            if saved_video_id and saved_video_id != video_id:
                st.warning(
                    f"⚠️ 이전 퀴즈는 다른 영상(`{saved_video_id}`)의 것입니다. 새로운 요약을 작성하고 퀴즈를 다시 생성해주세요."
                )

            # ===== 2️⃣ 자막 가져오기 섹션 (선택사항) =====
            st.divider()
            st.subheader("2️⃣ 자막 가져오기 (선택사항)")

            st.markdown(f"""
            **🤖 이 영상(`{video_id}`)의 자막을 자동으로 가져올 수 있습니다.**
            
            자막을 참고하여 아래 요약을 작성하거나, 자막을 그대로 학습 자료로 사용할 수 있습니다.
            """)

            # 자막 자동 가져오기 버튼
            col_sub1, col_sub2 = st.columns([2, 1])

            with col_sub1:
                fetch_clicked = st.button(
                    "🎬 자막 가져오기",
                    key=f"fetch_subtitle_{video_id}",
                    width="stretch",
                )

            with col_sub2:
                reset_subtitle_clicked = st.button(
                    "🔄 자막 초기화", key=f"reset_subtitle_{video_id}"
                )

            # 자막 가져오기 버튼 처리
            if fetch_clicked:
                # 자막 가져오기 중 플래그 설정 (URL 변경 감지에서 초기화 방지)
                st.session_state["fetching_transcript"] = True

                with st.spinner("자막을 가져오는 중..."):
                    result = get_youtube_transcript(video_id, language="id")

                if result["success"]:
                    # 자막을 별도 키에 저장 (읽기 전용 표시용)
                    subtitle_key = f"youtube_fetched_subtitle_{video_id}"
                    st.session_state[subtitle_key] = result["transcript"]

                    # 어떤 언어의 자막을 가져왔는지 표시
                    lang_used = result.get("language_used", "id")
                    if lang_used == "id":
                        st.success(
                            f"✅ 인도네시아어 자막을 성공적으로 가져왔습니다! ({len(result['transcript'])}자)"
                        )
                    else:
                        st.warning(
                            f"⚠️ 인도네시아어 자막이 없어 {lang_used} 자막을 가져왔습니다. ({len(result['transcript'])}자)"
                        )
                        st.info(
                            "💡 아래에서 자막을 확인하고 참고하여 요약을 작성하세요."
                        )
                else:
                    st.error(f"❌ {result['error']}")
                    st.info(
                        "💡 자막이 없는 경우 직접 영상을 시청하고 아래에 요약을 작성해주세요."
                    )

            # 초기화 버튼 처리
            if reset_subtitle_clicked:
                subtitle_key = f"youtube_fetched_subtitle_{video_id}"
                st.session_state.pop(subtitle_key, None)
                st.info("🔄 자막이 초기화되었습니다.")

            # 가져온 자막 표시 (읽기 전용, 가독성 개선됨)
            subtitle_key = f"youtube_fetched_subtitle_{video_id}"
            fetched_subtitle = st.session_state.get(subtitle_key, "")

            if fetched_subtitle:
                st.markdown("**📄 가져온 자막 (읽기 전용, 30초 단위로 문단 구분):**")
                st.text_area(
                    "자막 내용",
                    value=fetched_subtitle,
                    height=500,  # 200 → 500 (2.5배 확대)
                    disabled=True,
                    key=f"display_subtitle_{video_id}",
                    label_visibility="collapsed",
                )
                # 문단 수 계산 (빈 줄 기준)
                paragraph_count = fetched_subtitle.count("\n\n") + 1
                st.caption(
                    f"📊 자막 길이: {len(fetched_subtitle)}자 | 문단 수: {paragraph_count}개"
                )

            # ===== 3️⃣ 쉬운 글쓰기 + 문법 피드백 섹션 =====
            st.divider()
            st.subheader("3️⃣ 쉬운 글쓰기 + 문법 피드백")

            st.markdown("**아래 템플릿을 채워서 2~3문장만 작성해보세요.**")
            st.markdown(
                """
**채워 넣기 템플릿**
1) Video ini tentang ...
2) Pembicara mengatakan bahwa ...
3) Hal menarik adalah ...
"""
            )

            writing_key = f"youtube_guided_writing_{video_id}"
            youtube_writing_input = st.text_area(
                "✍️ 간단한 문장 작성 (인도네시아어)",
                height=200,
                placeholder="예: Video ini tentang kebiasaan sehat di sekolah.",
                key=writing_key,
            )

            feedback_key = f"youtube_writing_feedback_{video_id}"
            feedback_error_key = f"youtube_writing_feedback_error_{video_id}"

            if st.button(
                "✅ 첨삭 받기",
                width="stretch",
                key=f"youtube_writing_feedback_submit_{video_id}",
            ):
                if not youtube_writing_input.strip():
                    st.warning("먼저 간단한 문장을 작성해주세요.")
                else:
                    try:
                        transcript_for_feedback = (
                            fetched_subtitle[:2000] if fetched_subtitle else ""
                        )
                        prompt = safe_prompt_fill(
                            YOUTUBE_WRITING_FEEDBACK_PROMPT,
                            level=level,
                            transcript=transcript_for_feedback,
                            user_text=youtube_writing_input.strip(),
                        )
                        feedback = llm_json(prompt, model=gen_model)
                        st.session_state[feedback_key] = feedback
                        st.session_state.pop(feedback_error_key, None)
                        st.success("✅ 첨삭 완료!")
                        st.rerun()
                    except Exception as e:
                        st.session_state[feedback_error_key] = str(e)
                        st.error("❌ 첨삭 실패")

            if feedback_key in st.session_state:
                feedback = st.session_state.get(feedback_key, {})
                st.markdown("#### ✅ 교정된 문장")
                st.success(feedback.get("corrected_text", ""))

                fixes = feedback.get("fixes", [])
                if fixes:
                    st.markdown("#### 🔧 주요 수정 포인트")
                    for fix in fixes:
                        before = fix.get("before", "")
                        after = fix.get("after", "")
                        why_ko = fix.get("why_ko", "")
                        st.markdown(f"- **전:** {before} → **후:** {after}")
                        if why_ko:
                            st.caption(f"이유: {why_ko}")

                overall_comment = feedback.get("overall_comment_ko", "")
                if overall_comment:
                    st.markdown("#### 💬 코멘트")
                    st.info(overall_comment)

                tips = feedback.get("tips_ko", [])
                if tips:
                    st.markdown("#### ✅ 다음에 바로 적용할 팁")
                    for tip in tips:
                        st.markdown(f"- {tip}")

            # 퀴즈 생성 버튼 - 글쓰기 아래
            st.markdown("---")

            # 자막이 있으면 우선 사용, 없으면 작성문 사용
            text_for_quiz = fetched_subtitle or youtube_writing_input.strip()
            quiz_char_count = len(text_for_quiz)

            if quiz_char_count >= 50:
                # 버튼 key도 video_id를 포함시켜 URL마다 독립적으로
                quiz_btn_key = f"btn_generate_youtube_quiz_{video_id}"

                # 어떤 자료로 퀴즈를 생성하는지 표시
                if youtube_writing_input.strip():
                    btn_label = f"🎯 가져온 영상 자막으로 퀴즈 {num_questions}문항 생성"
                else:
                    btn_label = f"🎯 가져온 자막으로 퀴즈 {num_questions}문항 생성"

                if st.button(
                    btn_label, type="primary", key=quiz_btn_key, width="stretch"
                ):
                    # session_state에 플래그 및 현재 URL 정보 저장
                    st.session_state["start_quiz_generation"] = True
                    st.session_state["youtube_transcript"] = text_for_quiz
                    st.session_state["youtube_quiz_video_id"] = (
                        video_id  # 현재 비디오 ID 저장
                    )
                    st.session_state["youtube_current_url"] = (
                        youtube_url  # 현재 URL도 저장
                    )
                    # 이전 퀴즈 강제 초기화 (새 영상의 퀴즈 생성 보장)
                    st.session_state.pop("youtube_quiz", None)
                    st.session_state.pop("youtube_coach", None)
                    reset_mode_ephemeral("video")
                    RepeatLearningManager.reset("video")
                    st.rerun()
            else:
                quiz_btn_disabled_key = f"btn_generate_youtube_quiz_disabled_{video_id}"
                st.button(
                    f"🎯 퀴즈 {num_questions}문항 생성하기",
                    type="primary",
                    key=quiz_btn_disabled_key,
                    width="stretch",
                    disabled=True,
                )
                if fetched_subtitle:
                    st.caption(
                        "💡 자막을 가져왔으므로 바로 퀴즈를 생성할 수 있습니다. 또는 요약을 작성하세요."
                    )
                else:
                    st.caption(
                        "💡 자막을 가져오거나 요약을 작성하면 (최소 50자) 버튼이 활성화됩니다."
                    )
        else:
            st.error("❌ 올바른 YouTube URL이 아닙니다.")
            youtube_transcript_input = ""
    else:
        st.info("👆 위에 YouTube URL을 입력하거나 '📺 샘플 로드' 버튼을 눌러주세요.")
        youtube_transcript_input = ""

    # 퀴즈 생성 처리
    if st.session_state.get("start_quiz_generation"):
        st.divider()

        # 현재 생성 중인 영상 정보 표시
        generating_video_id = st.session_state.get("youtube_quiz_video_id", "unknown")
        generating_url = st.session_state.get("youtube_current_url", "")
        st.subheader(f"3️⃣ 퀴즈 생성 중... (영상 ID: `{generating_video_id}`)")

        if generating_url:
            st.caption(f"📹 URL: {generating_url}")

        saved_transcript = st.session_state.get("youtube_transcript", "")

        if saved_transcript:
            st.info(f"📝 요약 길이: {len(saved_transcript)}자")

            try:
                quiz_text = (
                    saved_transcript[:4000]
                    if len(saved_transcript) > 4000
                    else saved_transcript
                )
                with st.spinner(
                    f"영상 `{generating_video_id}`에 대한 퀴즈를 생성 중... (약 10초 소요)"
                ):
                    quiz, prompt = generate_quiz_with_checks(
                        quiz_text,
                        num_questions,
                        level,
                        gen_model,
                        debug,
                    )

                if debug:
                    with st.expander("🔍 DEBUG: QUIZ_PROMPT"):
                        st.code(prompt[:1000])

                st.session_state["youtube_quiz"] = quiz
                st.session_state.pop("youtube_coach", None)
                st.session_state.pop("start_quiz_generation")  # 플래그 제거
                st.success(f"✅ 퀴즈 생성 완료! (영상 ID: `{generating_video_id}`)")
                st.rerun()

            except Exception as e:
                st.error(f"❌ 퀴즈 생성 실패 (영상 ID: `{generating_video_id}`)")
                st.exception(e)
                st.session_state.pop(
                    "start_quiz_generation", None
                )  # 오류 시에도 플래그 제거

    # 퀴즈 표시 및 답안 입력
    youtube_quiz = st.session_state.get("youtube_quiz")

    # 현재 URL의 퀴즈인지 확인 (video_id 일치 여부)
    if youtube_quiz and youtube_url:
        current_video_id = extract_youtube_id(youtube_url)
        saved_video_id = st.session_state.get("youtube_quiz_video_id", "")

        # 비디오 ID가 다르면 퀴즈 무효화
        if current_video_id != saved_video_id:
            st.warning(
                f"⚠️ 표시된 퀴즈는 다른 영상(`{saved_video_id}`)의 것입니다. 현재 영상(`{current_video_id}`)에 대한 퀴즈를 생성하려면 위에서 요약을 작성하고 퀴즈 생성 버튼을 눌러주세요."
            )
            youtube_quiz = None
            st.session_state.pop("youtube_quiz", None)
            st.session_state.pop("youtube_coach", None)
            reset_mode_ephemeral("video")

    if youtube_quiz:
        st.divider()

        # 퀴즈가 어느 영상의 것인지 명확히 표시
        quiz_video_id = st.session_state.get("youtube_quiz_video_id", "unknown")
        st.subheader(f"3️⃣ 퀴즈 풀이 (영상 ID: `{quiz_video_id}`)")

        if youtube_quiz:
            questions = youtube_quiz.get("questions", [])

            if questions:
                with st.form("youtube_quiz_form"):
                    user_answers = {}

                    for q in questions:
                        qid = q.get("id")
                        st.markdown(f"**Q{qid}. {q.get('question', '')}**")

                        choices = q.get("choices", {})
                        opts = ["A", "B", "C", "D"]

                        # 초기에 아무것도 선택되지 않도록 index=None 설정
                        pick = st.radio(
                            "답 선택",
                            options=opts,
                            format_func=lambda k,
                            choices=choices: f"{k}. {choices.get(k, '')}",
                            key=f"youtube_q_{qid}",
                            index=None,
                            horizontal=True,
                        )
                        user_answers[str(qid)] = pick if pick else ""
                        st.divider()

                    submitted = st.form_submit_button("✅ 채점하기", type="primary")

                # 채점 및 코칭
                if submitted:
                    # 모든 답안이 선택되었는지 확인
                    empty_answers = [
                        qid for qid, ans in user_answers.items() if not ans
                    ]
                    if empty_answers:
                        st.error(
                            f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})"
                        )
                    else:
                        try:
                            condition_simple = condition or "B"
                            saved_transcript = st.session_state.get(
                                "youtube_transcript", ""
                            )

                            # 1) 순수 채점(점수/오답) 먼저 계산해서 저장
                            grading = grade_quiz(youtube_quiz, user_answers)
                            st.session_state["youtube_grading"] = {
                                "score": grading.score,
                                "total": grading.total,
                                "wrong_items": [
                                    wi.__dict__ for wi in grading.wrong_items
                                ],
                            }

                            st.session_state["youtube_user_answers"] = (
                                _normalize_user_answers(user_answers)
                            )

                            prompt = safe_prompt_fill(
                                COACH_PROMPT,
                                transcript=(
                                    saved_transcript[:4000]
                                    if saved_transcript and len(saved_transcript) > 4000
                                    else (saved_transcript or "")
                                ),
                                quiz_json=json.dumps(youtube_quiz, ensure_ascii=False),
                                user_answers_json=json.dumps(
                                    user_answers, ensure_ascii=False
                                ),
                                condition=condition_simple,
                            )

                            with st.spinner("채점 중..."):
                                # Structured Outputs 사용
                                coach = llm_structured(
                                    prompt, CoachResponse, model=gen_model
                                )
                                coach = sanitize_coach_structured(
                                    coach, youtube_quiz, user_answers
                                )

                            youtube_quiz = attach_explanations_to_quiz(
                                youtube_quiz, coach, user_answers
                            )
                            st.session_state["youtube_quiz"] = youtube_quiz

                            st.session_state["youtube_coach"] = coach

                            # 학습 기록 저장
                            wrong_items_analyzed = []
                            for item in coach.get("wrong_items", []):
                                q = next(
                                    (
                                        q
                                        for q in youtube_quiz.get("questions", [])
                                        if str(q.get("id")) == str(item.get("id"))
                                    ),
                                    {},
                                )
                                analyzed = WeaknessAnalyzer.analyze_wrong_answer(
                                    q,
                                    item.get("user_answer", ""),
                                    item.get("correct_answer", ""),
                                )
                                analyzed.update(item)
                                # ID 필드 명시적으로 설정 (SRS에 추가되도록)
                                q_id = str(q.get("id", item.get("id", "")))
                                analyzed["id"] = q_id
                                analyzed["question_id"] = q_id
                                wrong_items_analyzed.append(analyzed)

                            LearningHistoryManager.add_session(
                                {
                                    "source": "youtube",
                                    "level": level,
                                    "condition": condition,
                                    "score": coach.get("score", {}),
                                    "wrong_items": wrong_items_analyzed,
                                }
                            )

                            record_mode_result(
                                "video",
                                {
                                    "source": "video",
                                    "level": level,
                                    "condition": condition,
                                    "score": coach.get("score", {}),
                                    "wrong_items": wrong_items_analyzed,
                                    "quiz_questions": youtube_quiz.get("questions", []),
                                    "coach": coach,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )

                            st.success("✅ 채점 완료!")
                            st.rerun()

                        except Exception as e:
                            st.error("❌ 채점 실패")
                            st.exception(e)

                            # 디버그 모드에서 상세 정보 표시
                            if debug:
                                if "last_llm_response" in st.session_state:
                                    with st.expander("🔍 DEBUG: 오류 상세 정보"):
                                        st.json(st.session_state["last_llm_response"])

        # 코칭 결과 표시
        youtube_coach = st.session_state.get("youtube_coach")

        if youtube_coach:
            st.divider()

            # 결과가 어느 영상의 것인지 표시
            result_video_id = st.session_state.get("youtube_quiz_video_id", "unknown")
            st.markdown(f"### 🎓 학습 결과 (영상 ID: `{result_video_id}`)")

            # 점수 표시
            score = youtube_coach.get("score", {})
            correct = score.get("correct", 0)
            total = score.get("total", 5)
            percent = score.get("percent", 0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("정답 수", f"{correct}/{total}")
            col2.metric("정답률", f"{percent}%")
            col3.metric("컨디션", condition or "미설정")
            col4.metric("영상 ID", f"{result_video_id[:8]}...")

            # 오답 풀이 및 해설
            st.divider()
            st.markdown("#### 📝 문제 풀이 및 해설")

            youtube_quiz = st.session_state.get("youtube_quiz", {})
            questions = youtube_quiz.get("questions", [])
            wrong_items = youtube_coach.get("wrong_items", [])
            wrong_items_by_id = {str(item.get("id")): item for item in wrong_items}
            items_by_id = {
                str(item.get("id")): item for item in youtube_coach.get("items", [])
            }
            user_answers = st.session_state.get("youtube_user_answers", {})
            computed_wrong_items, _ = compute_wrong_items(questions, user_answers)
            wrong_items_for_repeat = merge_wrong_items_for_repeat(
                wrong_items, computed_wrong_items
            )
            all_correct = len(computed_wrong_items) == 0

            for q in questions:
                qid = str(q.get("id"))
                item = items_by_id.get(qid, {})
                wrong_item = wrong_items_by_id.get(qid, {})
                user_ans = user_answers.get(qid) or wrong_item.get("user_answer", "")
                render_question_result(
                    mode="youtube",
                    question=q,
                    item=item,
                    user_answer=user_ans,
                    wrong_item=wrong_item,
                    wrong_items_by_id=wrong_items_by_id,
                    key_prefix="youtube",
                    navigate_to_page_fn=navigate_to_page,
                )

            st.divider()
            render_quiz_takeaways(youtube_coach, wrong_items)

            # 취약 포인트
            st.divider()
            st.markdown("#### 🎯 취약 포인트")
            for wp in youtube_coach.get("weak_points_ko", []):
                st.markdown(f"- {wp}")

            # 내일 학습 플랜
            st.divider()
            st.markdown("#### 📅 내일 10분 학습 플랜")
            for step in youtube_coach.get("tomorrow_plan_10min_ko", []):
                st.markdown(f"- **{step.get('minute', '')}분**: {step.get('task', '')}")

            # Shadowing 문장
            st.markdown("#### 🗣️ Shadowing 연습")
            for s in youtube_coach.get("shadowing_sentences", []):
                st.markdown(f"**{s.get('id', '')}**")
                st.markdown(f"→ _{s.get('ko', '')}_")
                st.markdown("")

            # AI 학습 코치 (코치 결과 기반)
            if wrong_items and len(wrong_items) > 0:
                st.divider()
                render_ai_learning_coach(
                    wrong_items=wrong_items,
                    score_info={"correct": correct, "total": total, "percent": percent},
                    condition=condition,
                    key_prefix="youtube",
                )

            st.divider()

            # 반복 학습이 이미 진행 중인지 확인
            repeat_progress = RepeatLearningManager.get_progress("video")

            if is_debug_mode():
                st.caption(
                    f"[debug] mode=video all_correct={all_correct} "
                    f"wrong_count={len(computed_wrong_items)} "
                    f"repeat_active={repeat_progress['active']}"
                )

            st.markdown("#### 🔄 반복 학습")

            if repeat_progress["active"]:
                render_repeat_learning_ui("video", key_prefix="youtube")
            else:
                if not all_correct:
                    st.info(
                        f"💡 틀린 문제 {len(computed_wrong_items)}개를 모두 맞출 때까지 반복 학습할 수 있습니다!"
                    )

                    if st.button(
                        "🚀 틀린 문제 반복 학습 시작",
                        type="primary",
                        width="stretch",
                        key="youtube_start_repeat",
                    ):
                        # 취약점 분석 추가
                        analyzed_wrong = []
                        for item in wrong_items_for_repeat:
                            q_id = str(item.get("id"))
                            orig_q = next(
                                (q for q in questions if str(q.get("id")) == q_id), {}
                            )
                            analyzed = WeaknessAnalyzer.analyze_wrong_answer(
                                orig_q,
                                item.get("user_answer", ""),
                                item.get("correct_answer", ""),
                            )
                            analyzed["why_correct_ko"] = item.get("why_correct_ko", "")
                            analyzed["why_user_wrong_ko"] = item.get(
                                "why_user_wrong_ko", ""
                            )
                            analyzed_wrong.append(analyzed)

                        # 반복 학습 시작
                        RepeatLearningManager.start_repeat_learning(
                            "video", analyzed_wrong, questions
                        )
                        st.rerun()
                    entry_payload, entry_q_id = build_repeat_entry_shadowing_payload(
                        "video",
                        wrong_items_for_repeat,
                        questions,
                    )
                    if entry_payload and st.button(
                        "🗣️ 섀도잉 연습하기",
                        width="stretch",
                        key="youtube_repeat_shadowing_entry",
                    ):
                        st.session_state["shadowing_return_to"] = "results"
                        st.session_state["repeat_learning_resume"] = {
                            "mode": "video",
                            "q_id": entry_q_id,
                            "total_retries": 0,
                        }
                        st.session_state["force_review_tab"] = True
                        set_shadowing_payload_and_go(entry_payload, navigate_to_page)
                else:
                    st.success(
                        "🎉 모든 문제 정답입니다! 이제 유사 문제로 한 번 더 다져볼까요?"
                    )
                    if st.button(
                        "🔄 유사 문제로 반복 학습 시작",
                        type="primary",
                        width="stretch",
                        key="youtube_start_similar_repeat",
                    ):
                        with st.spinner("유사 문제를 생성 중..."):
                            similar_questions = build_similar_repeat_seed_questions(
                                "video",
                                questions,
                                model=gen_model,
                            )
                        RepeatLearningManager.start_custom(
                            "video", similar_questions, kind="similar"
                        )
                        st.rerun()

            # 학습 결과 페이지로 이동 버튼
            st.divider()
            if st.button(
                UI["btn_view_progress"],
                type="primary",
                width="stretch",
                key="youtube_goto_results",
            ):
                navigate_to_page("results")
    else:
        st.info("""
        📝 **요약을 작성해주세요!**
        
        1. 위에서 YouTube 영상을 시청하세요
        2. 영상 내용을 인도네시아어로 요약하세요 (최소 50자)
        3. 퀴즈 생성 버튼이 나타납니다
        
        💡 **팁**: 최소 5문장 이상 작성하면 좋은 퀴즈가 생성됩니다!
        """)

    # ✅ 페이지 하단 중앙 홈 버튼
    render_home_button_bottom(key="home_from_youtube")


def render_text_page():
    """텍스트 학습 페이지 렌더링"""
    gen_model = st.session_state.get("gen_model", "gpt-4o-mini")

    st.header("📄 텍스트로 학습하기")
    st.markdown("웹 링크를 입력하면 텍스트를 추출하여 학습 자료로 사용합니다.")

    render_learning_top_controls("text", navigate_to_page, key_prefix="text_top")

    # 1단계: 웹 링크 입력
    st.subheader("1️⃣ 웹 링크 입력")

    # 샘플 링크 표시
    with st.expander("📚 샘플 학습 자료", expanded=False):
        for source, link in SAMPLE_LINKS.items():
            st.markdown(f"[{source}]({link})")

    col_url1, col_url2 = st.columns([3, 1])

    with col_url1:
        text_url = st.text_input(
            "웹 페이지 URL",
            placeholder="https://...",
            key="text_url_input",
            help="VOA Indonesia, Wikisource 등의 URL을 입력하세요",
        )

    with col_url2:
        st.markdown("&nbsp;")  # 공백
        extract_btn = st.button(
            "🔍 추출", key="btn_extract_text", type="primary", width="stretch"
        )

    # URL 입력 안내
    if text_url and not extract_btn:
        st.info(
            "👉 URL을 입력했습니다. 위의 **'🔍 추출'** 버튼을 눌러 텍스트를 가져오세요."
        )

    # 추출 버튼 처리
    if extract_btn:
        if text_url:
            clean_url = normalize_url_input(text_url)

            with st.spinner(f"'{clean_url}'에서 텍스트를 추출 중..."):
                result = extract_text_from_url(clean_url)

            if result["success"]:
                st.session_state["extracted_text"] = result["text"]
                st.session_state["extracted_title"] = result["title"]
                st.session_state["current_source"] = f"Web: {clean_url}"
                st.session_state["current_text_url"] = clean_url  # 현재 URL 저장
                st.session_state.pop("text_quiz", None)
                st.session_state.pop("text_coach", None)
                reset_mode_ephemeral("text")
                RepeatLearningManager.reset("text")
                st.success(f"✅ 추출 완료: {result['title']}")
                st.rerun()
            else:
                st.error(f"❌ 추출 실패: {result['error']}")
        else:
            st.warning("⚠️ URL을 입력해주세요.")

    # 2단계: 추출된 텍스트
    extracted_text = st.session_state.get("extracted_text", "")
    extracted_title = st.session_state.get("extracted_title", "")
    current_text_url = st.session_state.get("current_text_url", "")

    if extracted_text:
        st.divider()
        st.subheader("2️⃣ 추출된 텍스트 (문단별로 구분됨)")

        if extracted_title:
            st.markdown(f"**📰 제목:** {extracted_title}")

        if current_text_url:
            st.caption(f"🔗 출처: {current_text_url}")

        # 전체 텍스트 표시 (포맷팅 적용됨)
        st.text_area(
            "인도네시아어 텍스트",
            value=extracted_text,
            height=500,  # 300 → 500 (확대)
            key="extracted_text_display",
            disabled=True,
        )

        # 문단 수 계산
        paragraph_count = extracted_text.count("\n\n") + 1
        st.caption(
            f"📊 텍스트 길이: {len(extracted_text)}자 | 문단 수: {paragraph_count}개"
        )

        # 퀴즈 생성 버튼
        st.markdown("---")

        if st.button(
            f"🎯 퀴즈 {num_questions}문항 생성하기",
            type="primary",
            key="btn_generate_text_quiz_main",
            width="stretch",
        ):
            st.session_state["start_text_quiz_generation"] = True
            st.rerun()
    else:
        st.info("👆 위에서 URL을 입력하고 '🔍 추출' 버튼을 눌러주세요.")

    # 퀴즈 생성 처리
    if st.session_state.get("start_text_quiz_generation"):
        st.divider()
        st.subheader("3️⃣ 퀴즈 생성 중...")

        saved_text = st.session_state.get("extracted_text", "")

        if saved_text:
            try:
                quiz_text = saved_text[:4000] if len(saved_text) > 4000 else saved_text
                with st.spinner("퀴즈를 생성 중... (약 10초 소요)"):
                    quiz, prompt = generate_quiz_with_checks(
                        quiz_text,
                        num_questions,
                        level,
                        gen_model,
                        debug,
                    )

                if debug:
                    with st.expander("🔍 DEBUG: QUIZ_PROMPT"):
                        st.code(prompt[:1000])

                st.session_state["text_quiz"] = quiz
                st.session_state.pop("text_coach", None)
                st.session_state.pop("start_text_quiz_generation")
                st.success("✅ 퀴즈 생성 완료!")
                st.rerun()

            except Exception as e:
                st.error("❌ 퀴즈 생성 실패")
                st.exception(e)
                st.session_state.pop("start_text_quiz_generation", None)

    # 퀴즈 표시 및 답안 입력
    text_quiz = st.session_state.get("text_quiz")

    if text_quiz:
        st.divider()
        st.subheader("3️⃣ 퀴즈 풀이")

        text_quiz = st.session_state.get("text_quiz")

        if text_quiz:
            questions = text_quiz.get("questions", [])

            if questions:
                with st.form("text_quiz_form"):
                    user_answers = {}

                    for q in questions:
                        qid = q.get("id")
                        st.markdown(f"**Q{qid}. {q.get('question', '')}**")

                        choices = q.get("choices", {})
                        opts = ["A", "B", "C", "D"]

                        # 초기에 아무것도 선택되지 않도록 index=None 설정
                        pick = st.radio(
                            "답 선택",
                            options=opts,
                            format_func=lambda k,
                            choices=choices: f"{k}. {choices.get(k, '')}",
                            key=f"text_q_{qid}",
                            index=None,
                            horizontal=True,
                        )
                        user_answers[str(qid)] = pick if pick else ""
                        st.divider()

                    submitted = st.form_submit_button("✅ 채점하기", type="primary")

                # 채점 및 코칭
                if submitted:
                    # 모든 답안이 선택되었는지 확인
                    empty_answers = [
                        qid for qid, ans in user_answers.items() if not ans
                    ]
                    if empty_answers:
                        st.error(
                            f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})"
                        )
                    else:
                        try:
                            condition_simple = condition or "B"
                            saved_text = st.session_state.get("extracted_text", "")

                            # 1) 순수 채점(점수/오답) 먼저 계산해서 저장
                            grading = grade_quiz(text_quiz, user_answers)
                            st.session_state["text_grading"] = {
                                "score": grading.score,
                                "total": grading.total,
                                "wrong_items": [
                                    wi.__dict__ for wi in grading.wrong_items
                                ],
                            }

                            st.session_state["text_user_answers"] = (
                                _normalize_user_answers(user_answers)
                            )

                            prompt = safe_prompt_fill(
                                COACH_PROMPT,
                                transcript=(
                                    saved_text[:4000]
                                    if saved_text and len(saved_text) > 4000
                                    else (saved_text or "")
                                ),
                                quiz_json=json.dumps(text_quiz, ensure_ascii=False),
                                user_answers_json=json.dumps(
                                    user_answers, ensure_ascii=False
                                ),
                                condition=condition_simple,
                            )

                            with st.spinner("채점 중..."):
                                # Structured Outputs 사용
                                coach = llm_structured(
                                    prompt, CoachResponse, model=gen_model
                                )
                                coach = sanitize_coach_structured(
                                    coach, text_quiz, user_answers
                                )

                            text_quiz = attach_explanations_to_quiz(
                                text_quiz, coach, user_answers
                            )
                            st.session_state["text_quiz"] = text_quiz

                            st.session_state["text_coach"] = coach

                            # 학습 기록 저장
                            wrong_items_analyzed = []
                            for item in coach.get("wrong_items", []):
                                q = next(
                                    (
                                        q
                                        for q in text_quiz.get("questions", [])
                                        if str(q.get("id")) == str(item.get("id"))
                                    ),
                                    {},
                                )
                                analyzed = WeaknessAnalyzer.analyze_wrong_answer(
                                    q,
                                    item.get("user_answer", ""),
                                    item.get("correct_answer", ""),
                                )
                                analyzed.update(item)
                                # ID 필드 명시적으로 설정 (SRS에 추가되도록)
                                q_id = str(q.get("id", item.get("id", "")))
                                analyzed["id"] = q_id
                                analyzed["question_id"] = q_id
                                wrong_items_analyzed.append(analyzed)

                            LearningHistoryManager.add_session(
                                {
                                    "source": "text",
                                    "level": level,
                                    "condition": condition,
                                    "score": coach.get("score", {}),
                                    "wrong_items": wrong_items_analyzed,
                                }
                            )

                            record_mode_result(
                                "text",
                                {
                                    "source": "text",
                                    "level": level,
                                    "condition": condition,
                                    "score": coach.get("score", {}),
                                    "wrong_items": wrong_items_analyzed,
                                    "quiz_questions": text_quiz.get("questions", []),
                                    "coach": coach,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )

                            st.success("✅ 채점 완료!")
                            st.rerun()

                        except Exception as e:
                            st.error("❌ 채점 실패")
                            st.exception(e)

                            # 디버그 모드에서 상세 정보 표시
                            if debug:
                                if "last_llm_response" in st.session_state:
                                    with st.expander("🔍 DEBUG: 오류 상세 정보"):
                                        st.json(st.session_state["last_llm_response"])

        # 코칭 결과 표시
        text_coach = st.session_state.get("text_coach")

        if text_coach:
            st.divider()
            st.markdown("### 🎓 학습 결과")

            # 점수 표시
            score = text_coach.get("score", {})
            correct = score.get("correct", 0)
            total = score.get("total", 5)
            percent = score.get("percent", 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("정답 수", f"{correct}/{total}")
            col2.metric("정답률", f"{percent}%")
            col3.metric("컨디션", condition or "미설정")

            # 오답 풀이 및 해설
            st.divider()
            st.markdown("#### 📝 문제 풀이 및 해설")

            text_quiz = st.session_state.get("text_quiz", {})
            questions = text_quiz.get("questions", [])
            wrong_items = text_coach.get("wrong_items", [])
            wrong_items_by_id = {str(item.get("id")): item for item in wrong_items}
            items_by_id = {
                str(item.get("id")): item for item in text_coach.get("items", [])
            }
            user_answers = st.session_state.get("text_user_answers", {})
            computed_wrong_items, _ = compute_wrong_items(questions, user_answers)
            wrong_items_for_repeat = merge_wrong_items_for_repeat(
                wrong_items, computed_wrong_items
            )
            all_correct = len(computed_wrong_items) == 0

            for q in questions:
                qid = str(q.get("id"))
                item = items_by_id.get(qid, {})
                wrong_item = wrong_items_by_id.get(qid, {})
                user_ans = user_answers.get(qid) or wrong_item.get("user_answer", "")
                render_question_result(
                    mode="text",
                    question=q,
                    item=item,
                    user_answer=user_ans,
                    wrong_item=wrong_item,
                    wrong_items_by_id=wrong_items_by_id,
                    key_prefix="text",
                    navigate_to_page_fn=navigate_to_page,
                )

            st.divider()
            render_quiz_takeaways(text_coach, wrong_items)

            # 취약 포인트
            st.divider()
            st.markdown("#### 🎯 취약 포인트")
            for wp in text_coach.get("weak_points_ko", []):
                st.markdown(f"- {wp}")

            # 내일 학습 플랜
            st.divider()
            st.markdown("#### 📅 내일 10분 학습 플랜")
            for step in text_coach.get("tomorrow_plan_10min_ko", []):
                st.markdown(f"- **{step.get('minute', '')}분**: {step.get('task', '')}")

            # Shadowing 문장
            st.markdown("#### 🗣️ Shadowing 연습")
            for s in text_coach.get("shadowing_sentences", []):
                st.markdown(f"**{s.get('id', '')}**")
                st.markdown(f"→ _{s.get('ko', '')}_")
                st.markdown("")

            # AI 학습 코치 (코치 결과 기반)
            if wrong_items and len(wrong_items) > 0:
                st.divider()
                render_ai_learning_coach(
                    wrong_items=wrong_items,
                    score_info={"correct": correct, "total": total, "percent": percent},
                    condition=condition,
                    key_prefix="text",
                )

            st.divider()

            # 반복 학습이 이미 진행 중인지 확인
            repeat_progress = RepeatLearningManager.get_progress("text")

            if is_debug_mode():
                st.caption(
                    f"[debug] mode=text all_correct={all_correct} "
                    f"wrong_count={len(computed_wrong_items)} "
                    f"repeat_active={repeat_progress['active']}"
                )

            st.markdown("#### 🔄 반복 학습")

            if repeat_progress["active"]:
                render_repeat_learning_ui("text", key_prefix="text")
            else:
                if not all_correct:
                    st.info(
                        f"💡 틀린 문제 {len(computed_wrong_items)}개를 모두 맞출 때까지 반복 학습할 수 있습니다!"
                    )

                    if st.button(
                        "🚀 틀린 문제 반복 학습 시작",
                        type="primary",
                        width="stretch",
                        key="text_start_repeat",
                    ):
                        # 취약점 분석 추가
                        analyzed_wrong = []
                        for item in wrong_items_for_repeat:
                            q_id = str(item.get("id"))
                            orig_q = next(
                                (q for q in questions if str(q.get("id")) == q_id), {}
                            )
                            analyzed = WeaknessAnalyzer.analyze_wrong_answer(
                                orig_q,
                                item.get("user_answer", ""),
                                item.get("correct_answer", ""),
                            )
                            analyzed["why_correct_ko"] = item.get("why_correct_ko", "")
                            analyzed["why_user_wrong_ko"] = item.get(
                                "why_user_wrong_ko", ""
                            )
                            analyzed_wrong.append(analyzed)

                        # 반복 학습 시작
                        RepeatLearningManager.start_repeat_learning(
                            "text", analyzed_wrong, questions
                        )
                        st.rerun()
                    entry_payload, entry_q_id = build_repeat_entry_shadowing_payload(
                        "text",
                        wrong_items_for_repeat,
                        questions,
                    )
                    if entry_payload and st.button(
                        "🗣️ 섀도잉 연습하기",
                        width="stretch",
                        key="text_repeat_shadowing_entry",
                    ):
                        st.session_state["shadowing_return_to"] = "results"
                        st.session_state["repeat_learning_resume"] = {
                            "mode": "text",
                            "q_id": entry_q_id,
                            "total_retries": 0,
                        }
                        st.session_state["force_review_tab"] = True
                        set_shadowing_payload_and_go(entry_payload, navigate_to_page)
                else:
                    st.success(
                        "🎉 모든 문제 정답입니다! 이제 유사 문제로 한 번 더 다져볼까요?"
                    )
                    if st.button(
                        "🔄 유사 문제로 반복 학습 시작",
                        type="primary",
                        width="stretch",
                        key="text_start_similar_repeat",
                    ):
                        with st.spinner("유사 문제를 생성 중..."):
                            similar_questions = build_similar_repeat_seed_questions(
                                "text",
                                questions,
                                model=gen_model,
                            )
                        RepeatLearningManager.start_custom(
                            "text", similar_questions, kind="similar"
                        )
                        st.rerun()

            # 학습 결과 페이지로 이동 버튼
            st.divider()
            if st.button(
                UI["btn_view_progress"],
                type="primary",
                width="stretch",
                key="text_goto_results",
            ):
                navigate_to_page("results")

    # ✅ 페이지 하단 중앙 홈 버튼
    render_home_button_bottom(key="home_from_text")


def render_shadowing_payload_block(navigate_to_page_fn, key_prefix: str = "shadowing"):
    payload = st.session_state.get("shadowing_payload")
    if not payload:
        return

    st.subheader("오답 섀도잉")
    return_to = st.session_state.get("shadowing_return_to", "results")

    col_back, col_done = st.columns(2)
    with col_back:
        if st.button("⬅️ 돌아가기", width="stretch", key=f"{key_prefix}_payload_back"):
            navigate_to_page_fn(return_to)
    with col_done:
        if st.button(
            "연습 완료",
            type="primary",
            width="stretch",
            key=f"{key_prefix}_payload_done",
        ):
            st.session_state.pop("shadowing_payload", None)
            navigate_to_page_fn(return_to)

    target_text = (payload.get("target_text") or "").strip()
    if target_text:
        st.markdown(f"## {target_text}")

    question_text = payload.get("question", "")
    correct_answer = payload.get("correct_answer", "")
    user_answer = payload.get("user_answer", "")
    rationale = payload.get("rationale", "")

    if question_text or correct_answer:
        st.markdown("**문맥**")
        if question_text:
            st.markdown(f"- 문제: {question_text}")
        if correct_answer:
            st.markdown(f"- 정답: {correct_answer}")
        if user_answer:
            st.markdown(f"- 내 답: {user_answer}")
        if rationale:
            st.markdown(f"- 해설: {rationale}")

    audio_ref = payload.get("audio_ref")
    if isinstance(audio_ref, dict):
        audio_ref = audio_ref.get("url") or audio_ref.get("path")
    if audio_ref:
        st.audio(audio_ref)

    speed = st.selectbox(
        "재생 속도",
        options=list(TTS_SPEED_OPTIONS.keys()),
        format_func=lambda x: TTS_SPEED_OPTIONS[x]["label"],
        index=2,
        key=f"{key_prefix}_payload_speed",
    )

    if target_text:
        render_tts_player(target_text, "", speed, f"{key_prefix}_payload")

    st.divider()


def render_speaking_hub(key_prefix: str = "speaking_hub"):
    mode_labels = {
        "audio": "Audio",
        "video": "Video",
        "text": "Text",
        "speaking": "Speaking",
        "all": "All",
    }

    selected_mode = st.radio(
        "모드 선택",
        options=["all", *MODES],
        format_func=lambda x: mode_labels.get(x, x.title()),
        horizontal=True,
        key=f"{key_prefix}_mode_selector",
    )

    speed = st.selectbox(
        "재생 속도",
        options=list(TTS_SPEED_OPTIONS.keys()),
        format_func=lambda x: TTS_SPEED_OPTIONS[x]["label"],
        index=2,
        key=f"{key_prefix}_speed_select",
    )

    if selected_mode == "all":
        for mode in MODES:
            st.markdown(f"#### {mode_labels.get(mode, mode.title())}")
            items = get_mode_state(mode).get("shadowing_items", [])
            render_mode_shadowing_items(mode, items, speed)
            st.divider()
    else:
        items = get_mode_state(selected_mode).get("shadowing_items", [])
        render_mode_shadowing_items(selected_mode, items, speed)

    st.divider()
    st.markdown("#### ✍️ 직접 입력하여 연습")

    custom_text = st.text_area(
        "인도네시아어 문장 입력",
        placeholder="Selamat pagi! Apa kabar?",
        height=100,
        key=f"{key_prefix}_custom_text",
    )

    if custom_text:
        render_tts_player(custom_text, "", speed, f"{key_prefix}_custom")


def render_speaking_page():
    st.subheader("🗣️ 섀도잉 연습")
    if "shadowing_payload" not in st.session_state:
        return_to = st.session_state.get("shadowing_return_to", "results")
        col_back, col_done = st.columns(2)
        with col_back:
            if st.button("⬅️ 돌아가기", width="stretch", key="speaking_page_back"):
                navigate_to_page(return_to)
        with col_done:
            if st.button(
                "연습 완료", type="primary", width="stretch", key="speaking_page_done"
            ):
                st.session_state.pop("shadowing_payload", None)
                navigate_to_page(return_to)
    render_shadowing_payload_block(navigate_to_page, key_prefix="speaking_page")
    render_speaking_hub(key_prefix="speaking_page")
    render_home_button_bottom(key="home_from_speaking")


def render_results_page():
    """학습 결과 페이지 렌더링"""

    st.markdown(
        f'<div class="section-title">{UI["progress_header"]}</div>',
        unsafe_allow_html=True,
    )

    tab_items = [
        ("dashboard", UI["tab_dashboard"]),
        ("review", UI["tab_review"]),
        ("speaking", UI["tab_speaking"]),
        ("quiz", UI["tab_quiz"]),
        ("memory", UI["tab_memory"]),
    ]

    if st.session_state.pop("force_review_tab", False):
        st.session_state["progress_tab"] = "review"

    if "progress_tab" not in st.session_state:
        st.session_state["progress_tab"] = "dashboard"

    active_tab = st.session_state["progress_tab"]

    st.markdown('<div id="progress-tabs-anchor"></div>', unsafe_allow_html=True)
    tab_cols = st.columns(5)
    for (tab_key, label), col in zip(tab_items, tab_cols):
        with col:
            if st.button(
                label,
                key=f"progress_tab_{tab_key}",
                type="primary" if active_tab == tab_key else "secondary",
                width="stretch",
            ):
                st.session_state["progress_tab"] = tab_key
                st.rerun()

    # ==========================================
    # 탭 1: 학습 대시보드
    # ==========================================
    if active_tab == "dashboard":
        st.subheader("📈 학습 대시보드")

        # 전체 통계 로드
        history_stats = LearningHistoryManager.get_stats()
        srs_stats = SpacedRepetitionSystem.get_stats()

        # 상단 통계 카드
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "총 학습 세션",
                f"{history_stats['total_sessions']}회",
                delta=f"+{history_stats['sessions_this_week']}회 (이번 주)",
            )

        with col2:
            delta_color = "normal" if history_stats["score_trend"] >= 0 else "inverse"
            st.metric(
                "평균 정답률",
                f"{history_stats['avg_score']}%",
                delta=f"{history_stats['score_trend']:+d}% (추세)",
                delta_color=delta_color,
            )

        with col3:
            st.metric(
                "연속 학습",
                f"{history_stats['streak_days']}일",
                delta="🔥 유지 중!" if history_stats["streak_days"] > 0 else None,
            )

        with col4:
            st.metric(
                "오늘 복습 예정",
                f"{srs_stats['due_today']}개",
                delta=f"전체 {srs_stats['total_items']}개 중",
            )

        st.divider()

        # 일별 학습 현황 (최근 7일)
        st.markdown("#### 📅 최근 7일 학습 현황")

        daily_stats = LearningHistoryManager.get_daily_stats(7)

        if any(d["sessions"] > 0 for d in daily_stats):
            # 간단한 텍스트 차트
            for day in daily_stats:
                date_label = day["date"][5:]  # MM-DD
                sessions = day["sessions"]
                score = day["avg_score"]

                bar = "█" * sessions + "░" * (
                    max(5, max(d["sessions"] for d in daily_stats)) - sessions
                )
                score_bar = (
                    "●" * (score // 10) + "○" * (10 - score // 10)
                    if score > 0
                    else "—" * 10
                )

                col_date, col_bar, col_score = st.columns([1, 2, 2])
                with col_date:
                    st.caption(date_label)
                with col_bar:
                    st.caption(f"세션: {bar} ({sessions})")
                with col_score:
                    st.caption(
                        f"점수: {score_bar} ({score}%)"
                        if score > 0
                        else "점수: — (없음)"
                    )
        else:
            st.info("아직 학습 기록이 없습니다. 퀴즈를 풀어보세요!")

        st.divider()

        # 취약점 분석
        st.markdown("#### 🎯 취약 카테고리 분석")

        weakness = LearningHistoryManager.get_weakness_analysis(10)

        if weakness["total_wrong"] > 0:
            st.caption(f"최근 10개 세션 기준, 총 {weakness['total_wrong']}개 오답 분석")

            # 클릭 가능한 카드로 변경
            for rec in weakness.get("recommendations", [])[:3]:
                cat_icon = rec.get("icon", "📌")
                cat_name = rec.get("name", rec.get("category", ""))
                cat_key = rec.get("category", "")
                count = rec.get("count", 0)
                activity = rec.get("activity", "")

                progress = (
                    count / weakness["total_wrong"]
                    if weakness["total_wrong"] > 0
                    else 0
                )

                # expander를 사용하여 클릭 가능한 카드 생성
                with st.expander(
                    f"{cat_icon} **{cat_name}**: {count}개 오답 ({progress * 100:.0f}%) — 클릭하여 상세 보기"
                ):
                    st.markdown("**💡 추천 학습 활동**")
                    st.info(activity)

                    # 해당 카테고리의 오답 근거 문장들 표시
                    cat_quotes = [
                        eq
                        for eq in weakness.get("evidence_quotes", [])
                        if eq.get("category") == cat_key
                    ]
                    if cat_quotes:
                        st.markdown(f"**📋 오답 근거 문장들 (총 {len(cat_quotes)}개)**")
                        for i, eq in enumerate(cat_quotes[:5], 1):  # 최대 5개만 표시
                            st.markdown(f'{i}. *"{eq.get("text", "")}"*')
                        if len(cat_quotes) > 5:
                            st.caption(f"... 외 {len(cat_quotes) - 5}개 더")

                    # 카테고리 정보 표시
                    cat_info = CEFR_CATEGORIES.get(cat_key, {})
                    if cat_info.get("description"):
                        st.markdown("**📖 카테고리 설명**")
                        st.caption(cat_info.get("description", ""))
        else:
            st.success("🎉 최근 오답이 없습니다! 훌륭해요!")

        st.divider()

        # SRS 카테고리별 통계
        st.markdown("#### 📚 카테고리별 학습 현황")

        cat_stats = SpacedRepetitionSystem.get_category_stats()

        # 디버그 정보 (개발 중에만 표시)
        if st.session_state.get("debug_mode_enabled", False):
            with st.expander("🔍 DEBUG: SRS 카테고리 통계"):
                st.write(f"총 카테고리 수: {len(cat_stats)}")
                st.json(cat_stats)
                srs_data = SpacedRepetitionSystem._load_data()
                st.write(f"SRS 총 항목 수: {len(srs_data.get('items', {}))}")
                st.json(list(srs_data.get("items", {}).values())[:3])  # 처음 3개만 표시

        if cat_stats and len(cat_stats) > 0:
            for cat_key, stats in cat_stats.items():
                cat_info = CEFR_CATEGORIES.get(cat_key, {"name": cat_key, "icon": "📌"})
                total = stats["total"]
                mastered = stats["mastered"]
                accuracy = (
                    int((stats["correct"] / stats["reviews"]) * 100)
                    if stats["reviews"] > 0
                    else 0
                )

                col_cat, col_progress, col_accuracy = st.columns([2, 3, 1])
                with col_cat:
                    st.markdown(f"{cat_info['icon']} **{cat_info['name']}**")
                with col_progress:
                    st.progress(
                        mastered / total if total > 0 else 0,
                        text=f"마스터: {mastered}/{total}",
                    )
                with col_accuracy:
                    st.caption(f"정확도: {accuracy}%")
        else:
            st.info(
                "💡 SRS에 등록된 항목이 없습니다. 퀴즈를 풀고 틀린 문제가 생기면 자동으로 등록됩니다!"
            )

    # ==========================================
    # 탭 2: 반복 학습 (틀린 문제 정답까지)
    # ==========================================
    if active_tab == "review":
        st.subheader("🔄 틀린 문제 반복 학습")
        st.info(
            "💡 틀린 문제를 모두 맞출 때까지 반복합니다. 유사 문제로 추가 연습도 가능합니다."
        )

        mode_labels = {
            "audio": "Audio",
            "video": "Video",
            "text": "Text",
            "speaking": "Speaking",
        }

        resume = st.session_state.get("repeat_learning_resume", {})
        resume_mode = resume.get("mode")

        for mode in MODES:
            mode_state = get_mode_state(mode)
            summary = mode_state.get("review_summary", {})
            last_result = mode_state.get("last_result") or {}
            wrong_items = mode_state.get("wrong_items", [])
            quiz_questions = last_result.get("quiz_questions", [])
            repeat_progress = RepeatLearningManager.get_progress(mode)

            st.markdown(f"#### {mode_labels.get(mode, mode.title())}")

            if summary.get("score"):
                score = summary.get("score", {})
                correct = score.get("correct", 0)
                total = score.get("total", 0)
                percent = score.get("percent", 0)
                col1, col2, col3 = st.columns(3)
                col1.metric("정답 수", f"{correct}/{total}")
                col2.metric("정답률", f"{percent}%")
                col3.metric("오답 수", f"{summary.get('wrong_count', 0)}")
                if summary.get("timestamp"):
                    st.caption(f"마지막 시도: {summary.get('timestamp')}")
            else:
                st.caption("아직 학습 결과가 없습니다.")

            if resume_mode == mode and repeat_progress["active"]:
                render_repeat_learning_ui(mode, key_prefix=f"review_{mode}")
                st.divider()
                continue

            if repeat_progress["active"]:
                render_repeat_learning_ui(mode, key_prefix=f"review_{mode}")
                st.divider()
                continue

            if not wrong_items:
                st.info("오답이 없습니다. 다른 모드에서 퀴즈를 풀어보세요.")
                st.divider()
                continue

            if not quiz_questions:
                st.warning("⚠️ 퀴즈 정보가 없어 반복 학습을 시작할 수 없습니다.")
                st.divider()
                continue

            st.success(f"✅ {len(wrong_items)}개의 틀린 문제가 있습니다.")
            with st.expander(
                f"🔍 {mode_labels.get(mode, mode.title())} 오답 미리보기",
                expanded=False,
            ):
                for item in wrong_items:
                    q_id = item.get("id")
                    orig_q = next(
                        (q for q in quiz_questions if str(q.get("id")) == str(q_id)), {}
                    )
                    question_text = orig_q.get(
                        "question", item.get("question", "문제 없음")
                    )
                    st.markdown(f"""
                    **Q{q_id}.** {question_text[:80]}...
                    - 내 답: {item.get("user_answer", "?")} ❌
                    - 정답: {item.get("correct_answer", "?")} ✅
                    """)

                    st.caption("💡 반복 학습에서 섀도잉 연습을 진행할 수 있습니다.")

            if st.button(
                "🚀 반복 학습 시작!",
                type="primary",
                width="stretch",
                key=f"review_start_repeat_{mode}",
            ):
                RepeatLearningManager.start_repeat_learning(
                    mode, wrong_items, quiz_questions
                )
                st.rerun()

            st.divider()

    # ==========================================
    # 탭 3: TTS 섀도잉
    # ==========================================
    if active_tab == "speaking":
        st.subheader("🗣️ 섀도잉 연습")
        render_shadowing_payload_block(navigate_to_page, key_prefix="results_speaking")
        render_speaking_hub(key_prefix="results_speaking")

    # ==========================================
    # 탭 4: 현재 세션 퀴즈 (기존 기능)
    # ==========================================
    if active_tab == "quiz":
        st.subheader("📝 현재 세션 퀴즈 풀이")

        # 교육적 가치 분석 결과 표시
        educational_analysis = st.session_state.get("educational_analysis")

        if educational_analysis:
            st.subheader("📋 교육적 가치 분석")

            with st.expander("📖 분석 결과 보기", expanded=True):
                st.markdown(
                    f"**주제:** {educational_analysis.get('main_topic', 'N/A')}"
                )
                st.markdown(
                    f"**교육 수준:** {educational_analysis.get('educational_level', 'N/A')}"
                )
                st.markdown(
                    f"**교육적 관련성:** {educational_analysis.get('relevance_score', 'N/A')}/10"
                )

                st.markdown("**주요 학습 포인트:**")
                for point in educational_analysis.get("key_learning_points", []):
                    st.markdown(f"- {point}")

                st.markdown("**콘텐츠 요약:**")
                st.write(educational_analysis.get("summary", ""))

        st.divider()

        # 퀴즈 생성 섹션
        st.subheader("📝 퀴즈 생성 및 풀이")

        # 현재 사용 가능한 텍스트 확인
        available_transcript = (
            st.session_state.get("audio_transcript")
            or st.session_state.get("youtube_transcript")
            or st.session_state.get("extracted_text")
        )

        if not available_transcript:
            st.info(
                "📌 먼저 '오디오 학습', 'YouTube 학습', 또는 '텍스트 학습' 탭에서 학습 자료를 준비해주세요."
            )
        else:
            current_source = st.session_state.get("current_source", "Unknown")
            st.caption(f"**출처:** {current_source}")

            # 퀴즈 생성 버튼
            if st.button("🎯 퀴즈 5문항 생성", type="primary", key="btn_generate_quiz"):
                try:
                    # 텍스트가 너무 길면 잘라서 사용
                    quiz_text = (
                        available_transcript[:4000]
                        if len(available_transcript) > 4000
                        else available_transcript
                    )
                    with st.spinner("퀴즈를 생성 중..."):
                        quiz, prompt = generate_quiz_with_checks(
                            quiz_text,
                            num_questions,
                            level,
                            gen_model,
                            debug,
                        )

                    if debug:
                        with st.expander("🔍 DEBUG: QUIZ_PROMPT (일부)"):
                            st.code(prompt[:1200])

                    st.session_state["quiz"] = quiz
                    st.session_state.pop("coach", None)  # 이전 코칭 결과 초기화
                    st.success("✅ 퀴즈 생성 완료!")

                except Exception as e:
                    st.error("❌ 퀴즈 생성 실패")
                    st.exception(e)

        # 퀴즈 표시 및 답안 입력
        quiz = st.session_state.get("quiz")

        if quiz:
            questions = quiz.get("questions", [])

            if not questions:
                st.warning("⚠️ 퀴즈 문제가 없습니다. 다시 생성해주세요.")
            else:
                st.markdown("### 📝 퀴즈 문제")

                with st.form("quiz_form"):
                    user_answers = {}

                    for q in questions:
                        qid = q.get("id")
                        st.markdown(f"**Q{qid}. {q.get('question', '')}**")

                        choices = q.get("choices", {})
                        opts = ["A", "B", "C", "D"]

                        # 초기에 아무것도 선택되지 않도록 index=None 설정
                        pick = st.radio(
                            f"답 선택 (Q{qid})",
                            options=opts,
                            format_func=lambda k,
                            choices=choices: f"{k}. {choices.get(k, '')}",
                            key=f"q_{qid}",
                            index=None,
                            horizontal=True,
                        )
                        user_answers[str(qid)] = pick if pick else ""

                        st.divider()

                    submitted = st.form_submit_button(
                        "✅ 채점하고 학습 플랜 받기", type="primary"
                    )

                # 채점 및 코칭
                if submitted:
                    # 모든 답안이 선택되었는지 확인
                    empty_answers = [
                        qid for qid, ans in user_answers.items() if not ans
                    ]
                    if empty_answers:
                        st.error(
                            f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})"
                        )
                    else:
                        try:
                            condition_simple = condition or "B"

                            # user_answers를 session_state에 저장 (payload에서 사용하기 위함)
                            # 1) 순수 채점(점수/오답) 먼저 계산해서 저장
                            grading = grade_quiz(quiz, user_answers)
                            st.session_state["tab4_grading"] = {
                                "score": grading.score,
                                "total": grading.total,
                                "wrong_items": [
                                    wi.__dict__ for wi in grading.wrong_items
                                ],
                            }

                            st.session_state["tab4_user_answers"] = user_answers

                            prompt = safe_prompt_fill(
                                COACH_PROMPT,
                                transcript=(
                                    available_transcript[:4000]
                                    if available_transcript
                                    and len(available_transcript) > 4000
                                    else (available_transcript or "")
                                ),
                                quiz_json=json.dumps(quiz, ensure_ascii=False),
                                user_answers_json=json.dumps(
                                    user_answers, ensure_ascii=False
                                ),
                                condition=condition_simple,
                            )

                            if debug:
                                with st.expander("🔍 DEBUG: COACH_PROMPT (일부)"):
                                    st.code(prompt[:1200])

                            with st.spinner("코칭 결과를 생성 중..."):
                                # Structured Outputs 사용
                                coach = llm_structured(
                                    prompt, CoachResponse, model=gen_model
                                )
                                coach = sanitize_coach_structured(
                                    coach, quiz, user_answers
                                )

                            st.session_state["coach"] = coach
                            st.success("✅ 채점 완료!")

                        except Exception as e:
                            st.error("❌ 채점 실패")
                            st.exception(e)

                            # 디버그 모드에서 상세 정보 표시
                            if debug:
                                if "last_llm_response" in st.session_state:
                                    with st.expander("🔍 DEBUG: 오류 상세 정보"):
                                        st.json(st.session_state["last_llm_response"])

        # 코칭 결과 표시
        coach = st.session_state.get("coach")

        if coach:
            st.divider()
            st.markdown("### 🎓 학습 결과 및 코칭")

            # 점수 표시
            score = coach.get("score", {})
            correct = score.get("correct", 0)
            total = score.get("total", 5)
            percent = score.get("percent", 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("정답 수", f"{correct}/{total}")
            col2.metric("정답률", f"{percent}%")
            col3.metric("컨디션", condition or "미설정")

            st.divider()

            # 취약 포인트
            st.markdown("#### 🎯 취약 포인트 3가지")
            for wp in coach.get("weak_points_ko", []):
                st.markdown(f"- {wp}")

            st.divider()

            # 내일 학습 플랜
            st.markdown("#### 📅 내일 10분 학습 플랜")
            for step in coach.get("tomorrow_plan_10min_ko", []):
                st.markdown(f"- **{step.get('minute', '')}분**: {step.get('task', '')}")

            st.divider()

            # Shadowing 문장
            st.markdown("#### 🗣️ Shadowing 연습 문장")
            for s in coach.get("shadowing_sentences", []):
                st.markdown(f"**{s.get('id', '')}**")
                st.markdown(f"→ _{s.get('ko', '')}_")
                st.markdown("")

            # 원본 JSON
            with st.expander("🔍 고급: 원본 JSON 보기"):
                st.json(coach)

            # 결과 저장
            st.divider()
            st.markdown("#### 💾 결과 저장")

            # user_answers를 session_state에서 가져오기 (form 스코프 문제 해결)
            saved_user_answers = st.session_state.get("tab4_user_answers", {})

            payload = {
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                "condition": condition,
                "source": st.session_state.get("current_source", "Unknown"),
                "transcript": available_transcript[:4000],
                "educational_analysis": educational_analysis,
                "quiz": quiz,
                "user_answers": saved_user_answers,
                "coach": coach,
            }

            fname = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            fpath = os.path.join(LOG_DIR, fname)

            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            st.download_button(
                label="📥 결과 JSON 다운로드",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=fname,
                mime="application/json",
            )

            st.caption(f"💾 로컬 저장: `{fpath}`")

    # ==========================================
    # 탭 5: SRS 간격 반복 복습
    # ==========================================
    if active_tab == "memory":
        st.subheader("📅 간격 반복 복습 (Spaced Repetition)")
        st.info(
            "💡 틀린 문제가 자동으로 SRS에 등록되어, 최적의 시간에 복습할 수 있습니다."
        )

        # SRS 통계
        srs_stats = SpacedRepetitionSystem.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 항목", srs_stats["total_items"])
        with col2:
            st.metric(
                "오늘 복습",
                srs_stats["due_today"],
                delta="예정" if srs_stats["due_today"] > 0 else None,
            )
        with col3:
            st.metric("마스터 완료", srs_stats["mastered"])
        with col4:
            st.metric("정확도", f"{srs_stats['avg_accuracy']}%")

        st.divider()

        # 오늘 복습할 항목
        due_items = SpacedRepetitionSystem.get_due_items(10)

        if due_items:
            st.markdown(f"### ⏰ 오늘 복습할 항목 ({len(due_items)}개)")

            for i, item in enumerate(due_items):
                content = item.get("content", {})
                question = content.get("question", "문제 없음")[:100]
                category = item.get("category", "unknown")
                level = item.get("level", 0)
                cat_info = CEFR_CATEGORIES.get(
                    category, {"icon": "📌", "name": category}
                )

                with st.expander(
                    f"{cat_info['icon']} {question}...", expanded=(i == 0)
                ):
                    st.markdown(f"**카테고리:** {cat_info['name']}")
                    st.markdown(f"**레벨:** {'⭐' * (level + 1)} ({level}/6)")
                    st.markdown(f"**복습 횟수:** {item.get('review_count', 0)}회")

                    if content.get("evidence_quote"):
                        st.markdown(f"**근거:** _{content['evidence_quote']}_")

                    # 선택지가 있는 경우
                    choices = content.get("choices", {})
                    correct_answer = content.get("correct_answer", "")

                    if choices:
                        st.markdown("**선택지:**")
                        for opt in ["A", "B", "C", "D"]:
                            if opt in choices:
                                mark = " ✅" if opt == correct_answer else ""
                                st.markdown(f"- {opt}. {choices[opt]}{mark}")

                    col_good, col_bad = st.columns(2)
                    with col_good:
                        if st.button("✅ 알았어요", key=f"srs_good_{item['id']}"):
                            SpacedRepetitionSystem.record_review(
                                item["id"], is_correct=True, quality=4
                            )
                            st.success("✅ 다음 복습은 나중에!")
                            st.rerun()
                    with col_bad:
                        if st.button("❌ 헷갈려요", key=f"srs_bad_{item['id']}"):
                            SpacedRepetitionSystem.record_review(
                                item["id"], is_correct=False, quality=2
                            )
                            st.warning("🔄 내일 다시 복습!")
                            st.rerun()
        else:
            st.success("🎉 오늘 복습할 항목이 없습니다! 훌륭해요!")

            if srs_stats["total_items"] == 0:
                st.info(
                    "💡 퀴즈에서 틀린 문제가 자동으로 SRS에 추가됩니다. 퀴즈를 풀어보세요!"
                )

        st.divider()

        # SRS 학습 곡선 설명
        with st.expander("📖 간격 반복 학습이란?", expanded=False):
            st.markdown("""
            **Spaced Repetition System (SRS)**는 기억을 최적화하는 학습 방법입니다.
            
            **작동 원리:**
            1. 처음 틀린 문제는 **1일 후** 복습
            2. 정답 시 간격 증가: 1일 → 3일 → 7일 → 14일 → 30일 → 60일
            3. 오답 시 간격 리셋: 다시 1일 후 복습
            
            **레벨 의미:**
            - ⭐ (레벨 0): 새로 추가됨
            - ⭐⭐ (레벨 1): 1회 정답
            - ⭐⭐⭐ (레벨 2): 2회 연속 정답
            - ⭐⭐⭐⭐⭐⭐ (레벨 5+): 마스터!
            
            **팁:** 매일 조금씩 복습하면 장기 기억에 더 잘 남습니다!
            """)

    if st.session_state.get("debug_mode_enabled", False):
        with st.expander("🔍 DEBUG: Mode State Keys"):
            for mode in MODES:
                mode_state = get_mode_state(mode)
                st.write(f"{mode}: {sorted(mode_state.keys())}")

    # ✅ 페이지 하단 중앙 홈 버튼
    render_home_button_bottom(key="home_from_results_bottom")


def render_settings_page():
    """설정 페이지 렌더링"""
    # 홈 버튼
    if st.button(UI["btn_home"], key="home_from_settings"):
        navigate_to_home()

    st.header("⚙️ 설정")
    st.markdown("앱 설정 및 로그 파일을 관리합니다.")

    st.divider()

    # ==========================================
    # 로그 파일 관리
    # ==========================================
    st.subheader("📁 로그 파일 관리")

    # 로그 통계 계산
    log_json = glob.glob(os.path.join(LOG_DIR, "log_*.json"))
    result_json = glob.glob(os.path.join(LOG_DIR, "result_*.json"))
    upload_wav = glob.glob(os.path.join(LOG_DIR, "upload_*.wav"))

    total_size = 0
    for file_list in [log_json, result_json, upload_wav]:
        for file in file_list:
            try:
                total_size += os.path.getsize(file)
            except:
                pass

    # 통계 표시
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("학습 로그", f"{len(log_json)}개", help="log_*.json 파일")
    col2.metric("결과 파일", f"{len(result_json)}개", help="result_*.json 파일")
    col3.metric("임시 오디오", f"{len(upload_wav)}개", help="upload_*.wav 파일")
    col4.metric(
        "총 용량", f"{total_size / 1024 / 1024:.1f} MB", help="전체 로그 폴더 용량"
    )

    st.divider()

    # ==========================================
    # 임시 오디오 파일 관리
    # ==========================================
    with st.expander("🎵 임시 오디오 파일 관리", expanded=False):
        st.markdown("""
        **임시 오디오 파일이란?**
        - 사용자가 업로드한 오디오의 임시 복사본입니다
        - ASR 처리 후에는 텍스트로 변환되어 JSON에 저장됩니다
        - 삭제해도 학습 기록에는 영향이 없습니다 ✅
        """)

        if upload_wav:
            st.markdown(f"**현재 임시 오디오 파일: {len(upload_wav)}개**")

            # 최근 5개만 표시
            display_count = min(5, len(upload_wav))
            sorted_wav = sorted(upload_wav, key=os.path.getmtime, reverse=True)

            for i, file in enumerate(sorted_wav[:display_count]):
                try:
                    file_size = os.path.getsize(file) / 1024  # KB
                    file_time = datetime.fromtimestamp(os.path.getmtime(file))
                    st.caption(
                        f"📄 {os.path.basename(file)} ({file_size:.1f} KB) - {file_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                except:
                    st.caption(f"📄 {os.path.basename(file)}")

            if len(upload_wav) > display_count:
                st.caption(f"... 외 {len(upload_wav) - display_count}개")

            st.divider()

            # 삭제 버튼
            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button(
                    "🗑️ 모든 임시 오디오 삭제", type="secondary", width="stretch"
                ):
                    deleted_count = 0
                    deleted_size = 0

                    for file in upload_wav:
                        try:
                            file_size = os.path.getsize(file)
                            os.remove(file)
                            deleted_count += 1
                            deleted_size += file_size
                        except Exception as e:
                            st.error(f"삭제 실패: {os.path.basename(file)} - {e}")

                    if deleted_count > 0:
                        st.success(
                            f"✅ {deleted_count}개 파일 삭제 완료 ({deleted_size / 1024 / 1024:.2f} MB 절약)"
                        )
                        st.rerun()
                    else:
                        st.warning("삭제된 파일이 없습니다.")

            with col_btn2:
                # 오래된 파일만 삭제 (7일 이전)
                old_wav = [
                    f
                    for f in upload_wav
                    if datetime.fromtimestamp(os.path.getmtime(f))
                    < datetime.now() - timedelta(days=7)
                ]

                if st.button(
                    f"🗑️ 7일 이전 파일 삭제 ({len(old_wav)}개)",
                    type="secondary",
                    width="stretch",
                    disabled=len(old_wav) == 0,
                ):
                    deleted_count = 0
                    deleted_size = 0

                    for file in old_wav:
                        try:
                            file_size = os.path.getsize(file)
                            os.remove(file)
                            deleted_count += 1
                            deleted_size += file_size
                        except Exception as e:
                            st.error(f"삭제 실패: {os.path.basename(file)} - {e}")

                    if deleted_count > 0:
                        st.success(
                            f"✅ {deleted_count}개 파일 삭제 완료 ({deleted_size / 1024 / 1024:.2f} MB 절약)"
                        )
                        st.rerun()
        else:
            st.info("💡 임시 오디오 파일이 없습니다.")

    # ==========================================
    # 학습 로그 파일 관리
    # ==========================================
    with st.expander("📝 학습 로그 파일 관리", expanded=False):
        st.markdown("""
        **학습 로그 파일이란?**
        - 학습 세션의 중간 기록입니다
        - 최종 결과는 `result_*.json`에 저장됩니다
        - 오래된 로그는 삭제해도 결과 파일에는 영향이 없습니다 ✅
        """)

        if log_json:
            st.markdown(f"**현재 학습 로그: {len(log_json)}개**")

            # 날짜별 그룹화
            logs_by_date = {}
            for file in log_json:
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(file))
                    date_key = mtime.strftime("%Y-%m-%d")
                    if date_key not in logs_by_date:
                        logs_by_date[date_key] = []
                    logs_by_date[date_key].append(file)
                except:
                    pass

            # 최근 3일치만 표시
            sorted_dates = sorted(logs_by_date.keys(), reverse=True)[:3]
            for date in sorted_dates:
                files = logs_by_date[date]
                st.caption(f"📅 {date}: {len(files)}개 파일")

            if len(logs_by_date) > 3:
                st.caption(f"... 외 {len(logs_by_date) - 3}일치")

            st.divider()

            # 삭제 버튼
            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                # 30일 이전 로그 삭제
                old_logs_30 = [
                    f
                    for f in log_json
                    if datetime.fromtimestamp(os.path.getmtime(f))
                    < datetime.now() - timedelta(days=30)
                ]

                if st.button(
                    f"🗑️ 30일 이전 로그 삭제 ({len(old_logs_30)}개)",
                    type="secondary",
                    width="stretch",
                    disabled=len(old_logs_30) == 0,
                ):
                    deleted_count = 0

                    for file in old_logs_30:
                        try:
                            os.remove(file)
                            deleted_count += 1
                        except Exception as e:
                            st.error(f"삭제 실패: {os.path.basename(file)} - {e}")

                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count}개 로그 삭제 완료")
                        st.rerun()

            with col_btn2:
                # 7일 이전 로그 삭제
                old_logs_7 = [
                    f
                    for f in log_json
                    if datetime.fromtimestamp(os.path.getmtime(f))
                    < datetime.now() - timedelta(days=7)
                ]

                if st.button(
                    f"🗑️ 7일 이전 로그 삭제 ({len(old_logs_7)}개)",
                    type="secondary",
                    width="stretch",
                    disabled=len(old_logs_7) == 0,
                ):
                    deleted_count = 0

                    for file in old_logs_7:
                        try:
                            os.remove(file)
                            deleted_count += 1
                        except Exception as e:
                            st.error(f"삭제 실패: {os.path.basename(file)} - {e}")

                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count}개 로그 삭제 완료")
                        st.rerun()
        else:
            st.info("💡 학습 로그 파일이 없습니다.")

    # ==========================================
    # 결과 파일 관리
    # ==========================================
    with st.expander("📊 결과 파일 관리", expanded=False):
        st.markdown("""
        **결과 파일이란?**
        - 퀴즈 결과 및 학습 기록이 저장된 중요한 파일입니다 ⚠️
        - 삭제하면 해당 학습 기록을 복구할 수 없습니다
        - 백업 후 삭제를 권장합니다
        """)

        if result_json:
            st.markdown(f"**현재 결과 파일: {len(result_json)}개**")

            # 최근 5개만 표시
            display_count = min(5, len(result_json))
            sorted_results = sorted(result_json, key=os.path.getmtime, reverse=True)

            for i, file in enumerate(sorted_results[:display_count]):
                try:
                    file_size = os.path.getsize(file) / 1024  # KB
                    file_time = datetime.fromtimestamp(os.path.getmtime(file))

                    col_file, col_download = st.columns([3, 1])

                    with col_file:
                        st.caption(
                            f"📄 {os.path.basename(file)} ({file_size:.1f} KB) - {file_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )

                    with col_download:
                        try:
                            with open(file, "r", encoding="utf-8") as f:
                                file_content = f.read()
                            st.download_button(
                                label="💾",
                                data=file_content,
                                file_name=os.path.basename(file),
                                mime="application/json",
                                key=f"download_result_{i}",
                                width="stretch",
                            )
                        except:
                            pass
                except:
                    st.caption(f"📄 {os.path.basename(file)}")

            if len(result_json) > display_count:
                st.caption(f"... 외 {len(result_json) - display_count}개")

            st.divider()

            # 백업 및 삭제
            st.warning(
                "⚠️ **주의**: 결과 파일을 삭제하면 복구할 수 없습니다. 백업 후 삭제하세요."
            )

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                # 모든 결과 파일 백업 (ZIP)
                if st.button(
                    "📦 모든 결과 백업 (ZIP)", type="primary", width="stretch"
                ):
                    import zipfile

                    backup_name = (
                        f"backup_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    )
                    backup_path = os.path.join(LOG_DIR, backup_name)

                    try:
                        with zipfile.ZipFile(backup_path, "w") as zipf:
                            for file in result_json:
                                zipf.write(file, os.path.basename(file))

                        st.success(f"✅ 백업 완료: {backup_name}")

                        # 다운로드 버튼 제공
                        with open(backup_path, "rb") as f:
                            st.download_button(
                                label="📥 백업 파일 다운로드",
                                data=f,
                                file_name=backup_name,
                                mime="application/zip",
                                key="download_backup",
                                width="stretch",
                            )
                    except Exception as e:
                        st.error(f"백업 실패: {e}")

            with col_btn2:
                # 30일 이전 결과 삭제
                old_results = [
                    f
                    for f in result_json
                    if datetime.fromtimestamp(os.path.getmtime(f))
                    < datetime.now() - timedelta(days=30)
                ]

                if st.button(
                    f"🗑️ 30일 이전 결과 삭제 ({len(old_results)}개)",
                    type="secondary",
                    width="stretch",
                    disabled=len(old_results) == 0,
                ):
                    deleted_count = 0

                    for file in old_results:
                        try:
                            os.remove(file)
                            deleted_count += 1
                        except Exception as e:
                            st.error(f"삭제 실패: {os.path.basename(file)} - {e}")

                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count}개 결과 파일 삭제 완료")
                        st.rerun()
        else:
            st.info("💡 결과 파일이 없습니다.")

    st.divider()

    # ==========================================
    # 앱 정보
    # ==========================================
    st.subheader("ℹ️ 앱 정보")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown(
            """
        **버전 정보**
        - 앱 버전: v1.0.0
        - Python: """
            + f"{os.sys.version.split()[0]}"
            + """
        - Streamlit: """
            + f"{st.__version__}"
            + """
        """
        )

    with col_info2:
        st.markdown(f"""
        **모델 정보**
        - ASR: Sparkplugx1904/whisper-base-id
        - LLM: {gen_model}
        - 타겟 언어: 인도네시아어
        """)

    st.divider()

    # ==========================================
    # 고급 설정
    # ==========================================
    with st.expander("🔧 고급 설정", expanded=False):
        st.markdown("**자동 정리 설정**")

        auto_clean_enabled = st.checkbox(
            "앱 시작 시 자동으로 오래된 임시 파일 삭제",
            value=False,
            help="7일 이상 된 임시 오디오 파일을 자동으로 삭제합니다",
        )

        if auto_clean_enabled:
            st.info(
                "💡 다음 앱 실행 시 자동 정리가 활성화됩니다. (현재 세션에서는 설정만 저장됩니다)"
            )

        st.divider()

        st.markdown("**캐시 초기화**")

        if st.button("🗑️ Streamlit 캐시 초기화", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ 캐시가 초기화되었습니다. ASR 모델이 다시 로드됩니다.")
            st.rerun()


# =====================================================
# 메인 라우터 - 페이지 네비게이션
# =====================================================

current_page = st.session_state.get("current_page", "home")
header_page = "" if current_page == "home" else current_page

render_header(
    header_page,
    navigate_to_page=navigate_to_page,
    navigate_to_home=navigate_to_home,
    logo_path=LOGO_PATH,
    nav_items=NAV_ITEMS,
    cta_label=UI["cta_make_it_bisa"],
)

if current_page == "home":
    render_home_page()
elif current_page == "audio":
    render_audio_page()
elif current_page == "youtube":
    render_youtube_page()
elif current_page == "text":
    render_text_page()
elif current_page == "results":
    render_results_page()
elif current_page == "speaking":
    render_speaking_page()
elif current_page == "settings":
    render_settings_page()
else:
    # 알 수 없는 페이지면 홈으로 리다이렉트
    st.session_state["current_page"] = "home"
    st.rerun()

render_footer()
