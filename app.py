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
from urllib.parse import urlparse, parse_qs
import streamlit as st
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline
import torchaudio
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import streamlit.components.v1 as components

# OpenAI 설정
load_dotenv()
from openai import OpenAI
client = OpenAI()

# Pydantic 모델 (Structured Outputs용)
from pydantic import BaseModel
from typing import Dict, List, Optional

# 프롬프트 불러오기
import prompts as P

missing = [name for name in ["QUIZ_PROMPT", "COACH_PROMPT", "EDUCATIONAL_ANALYSIS_PROMPT"] if not hasattr(P, name)]
if missing:
    raise ImportError(
        f"[prompts import check] Missing: {missing}\n"
        f"Loaded prompts.py from: {getattr(P, '__file__', 'unknown')}\n"
        f"Available names: {sorted([n for n in dir(P) if 'PROMPT' in n or 'CEFR' in n])}"
    )

QUIZ_PROMPT = P.QUIZ_PROMPT
COACH_PROMPT = P.COACH_PROMPT
EDUCATIONAL_ANALYSIS_PROMPT = P.EDUCATIONAL_ANALYSIS_PROMPT


# 상수 정의
APP_TITLE = "🎓 인도네시아어 학습 도구 (Indonesian Learning Tool)"
MODEL_ID = "Sparkplugx1904/whisper-base-id"
TARGET_SR = 16000
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 샘플 링크
SAMPLE_LINKS = {
    "Wikisource": "https://id.wikisource.org/wiki/Pangeran_Yang_Bahagia",
    "Wikinews (VOA)": "https://www.voaindonesia.com/a/guru-kreator-konten-mengapa-perlu-/7972528.html",
    "VOA Indonesia": "https://www.voaindonesia.com/a/guru-indonesia-dan-persepsi-tentang-as/6920962.html"
}

# =====================================================
# CEFR 기반 취약 카테고리 맵핑
# =====================================================

CEFR_CATEGORIES = {
    "vocabulary": {
        "name": "어휘 (Kosakata)",
        "icon": "📚",
        "keywords": ["arti", "makna", "kata", "kosakata", "어휘", "단어", "뜻", "의미"],
        "subcategories": {
            "basic_words": "기본 단어",
            "numbers": "숫자/수량 표현",
            "time_expressions": "시간 표현",
            "formal_vocabulary": "격식 어휘",
            "idioms": "관용구/숙어",
            "affixes": "접두사/접미사 (me-, ber-, -kan, -i)",
        }
    },
    "grammar": {
        "name": "문법 (Tata Bahasa)",
        "icon": "📝",
        "keywords": ["di-", "ter-", "me-", "ber-", "-kan", "접사", "수동", "문법", "시제"],
        "subcategories": {
            "tense": "시제 (sudah, akan, sedang)",
            "passive": "수동태 (di-, ter-)",
            "causative": "사역형 (-kan)",
            "prefix_suffix": "접사 체계",
            "reduplication": "반복어 (reduplikasi)",
            "conjunctions": "접속사",
            "prepositions": "전치사 (di, ke, dari)",
        }
    },
    "politeness": {
        "name": "경어/존칭 (Kesopanan)",
        "icon": "🎩",
        "keywords": ["bapak", "ibu", "pak", "bu", "존칭", "경어", "tolong", "mohon"],
        "subcategories": {
            "formal_register": "격식체",
            "honorifics": "존칭 (Bapak, Ibu, Pak, Bu)",
            "humble_forms": "겸양어",
            "request_forms": "요청 표현 (tolong, mohon)",
        }
    },
    "comprehension": {
        "name": "독해/이해 (Pemahaman)",
        "icon": "🔍",
        "keywords": ["utama", "pokok", "중심", "주제", "내용", "이해"],
        "subcategories": {
            "main_idea": "중심 내용 파악",
            "detail": "세부 정보",
            "inference": "추론",
            "context": "문맥 파악",
        }
    },
    "numbers": {
        "name": "숫자/수량 (Angka)",
        "icon": "🔢",
        "keywords": ["berapa", "jumlah", "angka", "숫자", "몇", "수량"],
        "subcategories": {
            "cardinal": "기수",
            "ordinal": "서수",
            "quantity": "수량 표현",
        }
    },
    "time": {
        "name": "시간 표현 (Waktu)",
        "icon": "⏰",
        "keywords": ["kapan", "waktu", "tanggal", "jam", "시간", "날짜", "언제"],
        "subcategories": {
            "clock_time": "시계 시간",
            "date": "날짜",
            "duration": "기간",
        }
    }
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
    }
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


def read_wav_resample(path: str, target_sr: int = 16000):
    """
    WAV 파일을 읽고 목표 샘플링 레이트로 리샘플링합니다.
    
    Args:
        path: WAV 파일 경로
        target_sr: 목표 샘플링 레이트 (기본값: 16000 Hz)
    
    Returns:
        (audio, sr): numpy 배열과 샘플링 레이트
    """
    audio, sr = sf.read(path)
    
    # 스테레오 -> 모노 변환
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    audio = audio.astype(np.float32)
    
    # 리샘플링
    if sr != target_sr:
        t = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        t = torchaudio.functional.resample(t, sr, target_sr)
        audio = t.squeeze(0).numpy()
        sr = target_sr
    
    return audio, sr


def transcribe_audio(asr_pipe, wav_path: str) -> str:
    """
    오디오 파일을 텍스트로 변환합니다.
    
    Args:
        asr_pipe: ASR 파이프라인
        wav_path: WAV 파일 경로
    
    Returns:
        str: 변환된 텍스트
    """
    audio, sr = read_wav_resample(wav_path, TARGET_SR)
    result = asr_pipe(
        {"array": audio, "sampling_rate": sr},
        generate_kwargs={"task": "transcribe", "language": "indonesian"},
        chunk_length_s=20,
        stride_length_s=3,
    )
    return result["text"].strip()


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
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
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
    sentence_pattern = r'([^.!?]+[.!?]+)'
    sentences = re.findall(sentence_pattern, text)
    
    # 패턴에 매칭되지 않은 나머지 텍스트 처리
    remaining = re.sub(sentence_pattern, '', text).strip()
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


def extract_text_from_url(url: str) -> dict:
    """
    URL에서 텍스트를 추출합니다.
    
    Args:
        url: 웹 페이지 URL
    
    Returns:
        dict: {"success": bool, "text": str, "title": str, "error": str}
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 제목 추출
        title = ""
        if soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.title:
            title = soup.title.string
        
        # 본문 추출 (일반적인 콘텐츠 태그들)
        # script, style 태그 제거
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # 본문 텍스트 추출
        text = ""
        
        # VOA 등의 뉴스 사이트
        article = soup.find('article') or soup.find('div', class_=re.compile('article|content|post|entry'))
        if article:
            text = article.get_text(separator='\n', strip=True)
        else:
            # 일반적인 경우
            text = soup.get_text(separator='\n', strip=True)
        
        # 빈 줄 제거 및 정리
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        raw_text = '\n'.join(lines)
        
        # 가독성 개선을 위한 포맷팅
        formatted_text = format_text_readable(raw_text, lines_per_paragraph=5)
        
        return {
            "success": True,
            "text": formatted_text,
            "title": title,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "title": "",
            "error": str(e)
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
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
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
                "error": None
            }
        
        except NoTranscriptFound:
            # 인도네시아어 자막이 없으면 영어로 시도
            try:
                fetched = api.fetch(video_id, languages=['en'])
                formatted_text = format_transcript_readable(fetched)
                
                return {
                    "success": True,
                    "transcript": formatted_text,
                    "language_used": "en (영어 자막)",
                    "error": None
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
                        "error": f"인도네시아어/영어 자막이 없습니다. 사용 가능한 언어: {', '.join(available)}"
                    }
                except:
                    return {
                        "success": False,
                        "transcript": "",
                        "language_used": "",
                        "error": "인도네시아어 자막을 찾을 수 없습니다."
                    }
    
    except TranscriptsDisabled:
        return {
            "success": False,
            "transcript": "",
            "language_used": "",
            "error": "이 영상은 자막이 비활성화되어 있습니다."
        }
    except Exception as e:
        return {
            "success": False,
            "transcript": "",
            "language_used": "",
            "error": f"자막 가져오기 실패: {str(e)}"
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
        scrolling=False
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
            "start_audio_quiz_generation"
        ]
    elif source_type == "youtube":
        keys_to_remove = [
            "youtube_transcript",
            "youtube_quiz",
            "youtube_coach",
            "start_quiz_generation",
            "youtube_quiz_video_id",
            "youtube_current_url"
        ]
    elif source_type == "text":
        keys_to_remove = [
            "extracted_text",
            "extracted_title",
            "text_quiz",
            "text_coach",
            "start_text_quiz_generation"
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
            {"role": "system", "content": "You must output ONLY valid JSON. No extra text, no markdown, no code blocks."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"}  # JSON mode 활성화
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
        
        st.warning("💡 퀴즈 생성을 다시 시도해주세요. 문제가 계속되면 텍스트 길이를 줄이거나 다른 자료를 사용해보세요.")
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
    choice_notes_ko: ChoiceNotes
    evidence_quote: str

class CoachResponse(BaseModel):
    """채점 및 코칭 전체 응답"""
    items: List[ExplainItem]
    weak_points_ko: List[str]
    tomorrow_plan_10min_ko: List[TomorrowPlanStep]
    shadowing_sentences: List[ShadowingSentence]
    
def llm_structured(prompt: str, response_model, model: str = "gpt-4o-mini"):
    """OpenAI Structured Outputs를 사용하여 스키마에 맞는 응답을 받습니다."""
    # ... 함수 본문

# =====================================================
# 3-3. 취약점 분석 시스템
# =====================================================

class WeaknessAnalyzer:
    """CEFR 기반 취약 카테고리 분석기"""
    
    @staticmethod
    def categorize_question(question: dict) -> tuple:
        """문제를 카테고리로 분류"""
        q_text = (question.get("question", "") + " " + str(question.get("choices", {}))).lower()
        
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
    def analyze_wrong_answer(question: dict, user_answer: str, correct_answer: str) -> dict:
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
                evidence_quotes.append({
                    "text": item["evidence_quote"],
                    "category": cat,
                    "question_id": item.get("question_id")
                })
        
        # 가장 취약한 카테고리
        primary_weakness = max(category_counts, key=category_counts.get) if category_counts else None
        
        # 추천 학습 활동 생성
        recommendations = []
        for cat, count in category_counts.items():
            if count >= 1:
                cat_info = CEFR_CATEGORIES.get(cat, {})
                recommendations.append({
                    "category": cat,
                    "icon": cat_info.get("icon", "📌"),
                    "name": cat_info.get("name", cat),
                    "count": count,
                    "message": f"{cat_info.get('name', cat)} 영역에서 {count}개 오답",
                    "activity": WeaknessAnalyzer._get_activity_recommendation(cat)
                })
        
        # count 순으로 정렬
        recommendations.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "total_wrong": len(wrong_items),
            "category_breakdown": category_counts,
            "subcategory_breakdown": subcategory_counts,
            "primary_weakness": primary_weakness,
            "evidence_quotes": evidence_quotes,
            "recommendations": recommendations
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
    
    SESSION_KEY = "repeat_learning_state"
    
    @classmethod
    def init_state(cls):
        """세션 상태 초기화"""
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = {
                "wrong_queue": [],        # 틀린 문제 대기열
                "current_question": None, # 현재 풀고 있는 문제
                "retry_count": {},        # 문제별 재시도 횟수
                "completed": [],          # 완료된 문제
                "total_retries": 0,       # 총 시도 횟수
                "active": False,          # 반복 학습 모드 활성화
            }
    
    @classmethod
    def start_repeat_learning(cls, wrong_items: list, quiz_questions: list):
        """반복 학습 시작"""
        cls.init_state()
        state = st.session_state[cls.SESSION_KEY]
        
        # 초기화
        state["wrong_queue"] = []
        state["completed"] = []
        state["retry_count"] = {}
        state["total_retries"] = 0
        state["active"] = True
        
        # quiz_questions를 딕셔너리로 변환
        q_dict = {str(q.get("id")): q for q in quiz_questions}
        
        for item in wrong_items:
            q_id = str(item.get("id"))
            full_question = q_dict.get(q_id, {})
            
            # 원본 문제 정보에 오답 정보 추가
            question_data = {
                **full_question,
                "user_wrong_answer": item.get("user_answer", ""),
                "evidence_quote": item.get("evidence_quote", full_question.get("evidence_quote", "")),
                "why_correct_ko": item.get("why_correct_ko", ""),
                "why_user_wrong_ko": item.get("why_user_wrong_ko", ""),
                "category": item.get("category", "comprehension"),
            }
            
            state["wrong_queue"].append(question_data)
            state["retry_count"][q_id] = 0
    
    @classmethod
    def get_next_question(cls) -> Optional[dict]:
        """다음 풀어야 할 문제 반환"""
        cls.init_state()
        state = st.session_state[cls.SESSION_KEY]
        
        if state["wrong_queue"]:
            state["current_question"] = state["wrong_queue"][0]
            return state["current_question"]
        return None
    
    @classmethod
    def check_answer(cls, user_answer: str) -> tuple:
        """답안 확인"""
        cls.init_state()
        state = st.session_state[cls.SESSION_KEY]
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
            state["wrong_queue"] = [q for q in state["wrong_queue"] if str(q.get("id")) != q_id]
            state["completed"].append({
                **current,
                "retries_needed": state["retry_count"][q_id]
            })
            state["current_question"] = None
        
        return is_correct, result
    
    @classmethod
    def replace_with_similar(cls, similar_question: dict):
        """현재 문제를 유사 문제로 교체"""
        cls.init_state()
        state = st.session_state[cls.SESSION_KEY]
        
        if state["wrong_queue"]:
            original_id = state["wrong_queue"][0].get("id")
            similar_question["original_id"] = original_id
            similar_question["is_similar"] = True
            state["wrong_queue"][0] = similar_question
            state["current_question"] = similar_question
    
    @classmethod
    def get_progress(cls) -> dict:
        """진행 상황 반환"""
        cls.init_state()
        state = st.session_state[cls.SESSION_KEY]
        
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
    def is_complete(cls) -> bool:
        """모든 문제 완료 여부"""
        cls.init_state()
        state = st.session_state[cls.SESSION_KEY]
        return len(state["wrong_queue"]) == 0 and len(state["completed"]) > 0
    
    @classmethod
    def reset(cls):
        """상태 초기화"""
        if cls.SESSION_KEY in st.session_state:
            st.session_state[cls.SESSION_KEY] = {
                "wrong_queue": [],
                "current_question": None,
                "retry_count": {},
                "completed": [],
                "total_retries": 0,
                "active": False,
            }


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
                with open(SRS_DATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"items": {}, "stats": {"total_reviews": 0}}
    
    @staticmethod
    def _save_data(data: dict):
        """SRS 데이터 저장"""
        with open(SRS_DATA_FILE, 'w', encoding='utf-8') as f:
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
                "level": 0,              # 복습 레벨 (0-6)
                "ease_factor": 2.5,      # 난이도 계수
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
            item["ease_factor"] = max(1.3, item["ease_factor"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
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
            "avg_accuracy": int((total_correct / total_reviews) * 100) if total_reviews > 0 else 0,
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
                category_stats[cat] = {"total": 0, "mastered": 0, "reviews": 0, "correct": 0}
            
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
                with open(LEARNING_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    @staticmethod
    def _save_history(history: list):
        """학습 기록 저장"""
        with open(LEARNING_HISTORY_FILE, 'w', encoding='utf-8') as f:
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
                }
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
            1 for s in history
            if datetime.fromisoformat(s.get("timestamp", "2000-01-01")) > week_ago
        )
        
        # 점수 추세
        if len(history) >= 5:
            recent_avg = sum(s.get("score", {}).get("percent", 0) for s in history[-5:]) / 5
            older_avg = sum(s.get("score", {}).get("percent", 0) for s in history[-10:-5]) / 5 if len(history) >= 10 else avg_score
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
        
        while current_date in dates or (current_date == today and (today - timedelta(days=1)) in dates):
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
            result.append({
                "date": date,
                "sessions": d["sessions"],
                "questions": d["questions"],
                "correct": d["correct"],
                "avg_score": int(d["score_sum"] / d["sessions"]) if d["sessions"] > 0 else 0
            })
        
        return result


# =====================================================
# 3-7. 유사 문제 생성
# =====================================================

SIMILAR_QUESTION_PROMPT = """당신은 인도네시아어 교육 전문가입니다.

다음 원본 문제와 **유사하지만 다른** 문제를 1개 생성해주세요.

**원본 문제:**
- 문제: {question}
- 카테고리: {category}
- 정답: {correct_answer}
- 원문 근거: {evidence_quote}

**요구사항:**
1. 같은 카테고리({category})와 난이도 유지
2. 같은 문법/어휘 개념을 테스트하되, 다른 문장/상황 사용
3. 원문 근거의 다른 부분이나 비슷한 패턴 활용
4. 문제와 선택지는 한국어로 작성
5. evidence_quote는 인도네시아어 원문

**반드시 아래 JSON 형식으로만 응답 (다른 텍스트 없이):**
{{
    "id": 99,
    "question": "새로운 문제 (한국어)",
    "category": "{category}",
    "choices": {{
        "A": "선택지 A",
        "B": "선택지 B",
        "C": "선택지 C",
        "D": "선택지 D"
    }},
    "answer": "정답(A/B/C/D 중 하나만)",
    "evidence_quote": "근거 문장 (인도네시아어)",
    "explanation": "정답 해설 (한국어)"
}}
"""

def generate_similar_question(original_question: dict, model: str = "gpt-4o-mini") -> Optional[dict]:
    """원본 문제와 유사한 새 문제 생성"""
    
    category, _ = WeaknessAnalyzer.categorize_question(original_question)
    cat_info = CEFR_CATEGORIES.get(category, {})
    
    prompt = SIMILAR_QUESTION_PROMPT.format(
        question=original_question.get("question", ""),
        category=f"{cat_info.get('name', category)} ({category})",
        correct_answer=original_question.get("answer", ""),
        evidence_quote=original_question.get("evidence_quote", "원문 없음"),
    )
    
    try:
        result = llm_json(prompt, model=model)
        result["is_similar"] = True
        result["original_id"] = original_question.get("id")
        result["category"] = category
        return result
    except Exception as e:
        st.error(f"유사 문제 생성 실패: {e}")
        return None


# =====================================================
# 3-8. TTS 섀도잉 기능
# =====================================================

def render_tts_player(text: str, translation: str = "", speed: str = "normal", key_suffix: str = ""):
    """
    TTS 재생 플레이어 렌더링 (브라우저 내장 Web Speech API 사용)
    
    Args:
        text: 읽을 텍스트 (인도네시아어)
        translation: 한국어 번역
        speed: 재생 속도 키
        key_suffix: 고유 키 접미사
    """
    rate = TTS_SPEED_OPTIONS.get(speed, {}).get("rate", 1.0)
    
    # 텍스트 표시
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 1rem; border-radius: 12px; margin: 0.5rem 0;
                border-left: 4px solid #667eea;">
        <p style="font-size: 1.1rem; color: #1e3c72; margin-bottom: 0.5rem; font-weight: 500;">
            🇮🇩 {text}
        </p>
        {f'<p style="color: #666; font-size: 0.9rem; margin: 0;">🇰🇷 {translation}</p>' if translation else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript TTS 버튼
    button_id = abs(hash(text + key_suffix)) % 1000000
    
    # HTML/JS로 TTS 구현
    components.html(f"""
    <div style="margin: 0.5rem 0;">
        <button onclick="speakText_{button_id}()" 
                style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; border: none; padding: 0.6rem 1.2rem;
                       border-radius: 25px; cursor: pointer; font-size: 0.9rem;
                       box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
                       transition: transform 0.2s, box-shadow 0.2s;"
                onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 15px rgba(102, 126, 234, 0.4)';"
                onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 10px rgba(102, 126, 234, 0.3)';">
            🔊 재생 ({TTS_SPEED_OPTIONS[speed]['label']})
        </button>
        <button onclick="stopSpeech()" 
                style="background: #dc3545; color: white; border: none; 
                       padding: 0.6rem 1rem; border-radius: 25px; cursor: pointer;
                       font-size: 0.9rem; margin-left: 0.5rem;">
            ⏹ 정지
        </button>
    </div>
    <script>
        function speakText_{button_id}() {{
            window.speechSynthesis.cancel();
            const text = `{text.replace('`', "'")}`;
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'id-ID';
            utterance.rate = {rate};
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
        }}
        function stopSpeech() {{
            window.speechSynthesis.cancel();
        }}
    </script>
    """, height=70)


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
            key="shadowing_speed_select"
        )
    
    # 틀린 문제 근거 문장 (우선 표시)
    wrong_items = coach_result.get("wrong_items", [])
    evidence_quotes = [item.get("evidence_quote", "") for item in wrong_items if item.get("evidence_quote")]
    
    if evidence_quotes:
        st.markdown("#### 📌 틀린 문제 근거 문장")
        st.caption("오답과 관련된 원문을 집중적으로 연습하세요.")
        
        for i, quote in enumerate(evidence_quotes):
            with st.expander(f"🔴 오답 근거 {i+1}", expanded=(i == 0)):
                why_correct = wrong_items[i].get("why_correct_ko", "") if i < len(wrong_items) else ""
                render_tts_player(
                    text=quote,
                    translation=why_correct[:100] + "..." if len(why_correct) > 100 else why_correct,
                    speed=speed,
                    key_suffix=f"evidence_{i}"
                )
    
    # 일반 섀도잉 문장
    shadowing_sentences = coach_result.get("shadowing_sentences", [])
    if shadowing_sentences:
        st.markdown("#### 📝 추가 연습 문장")
        
        for i, sentence in enumerate(shadowing_sentences):
            with st.expander(f"연습 {i+1}", expanded=False):
                text = sentence.get("id", "") if isinstance(sentence, dict) else str(sentence)
                translation = sentence.get("ko", "") if isinstance(sentence, dict) else ""
                render_tts_player(
                    text=text,
                    translation=translation,
                    speed=speed,
                    key_suffix=f"shadow_{i}"
                )
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
                {"role": "system", "content": "You are an Indonesian language learning coach. Return structured output that matches the schema."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_model,
            temperature=0.3,
        )
        
        parsed_response = completion.choices[0].message.parsed
        
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
            wrong_items.append({
                "id": int(qid) if qid.isdigit() else qid,
                "user_answer": user,
                "correct_answer": correct,
            })
    
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
            
            wrong_items.append({
                "id": item.get("id"),
                "user_answer": user_ans,
                "correct_answer": correct_ans,
                "why_correct_ko": item.get("correct_explain_ko", ""),
                "why_user_wrong_ko": item.get("wrong_reason_ko", ""),
                "evidence_quote": item.get("evidence_quote", ""),
                "choices_explanation": item.get("choice_notes_ko", {})
            })
    
    coach["wrong_items"] = wrong_items
    
    return coach


# =====================================================
# 5. Streamlit UI
# =====================================================

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ========== 커스텀 CSS ==========
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a8d8ea;
        margin-bottom: 1rem;
    }
    
    /* 기능 카드 */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* 퀴즈 카드 */
    .quiz-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .quiz-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* 결과 카드 */
    .result-correct {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .result-incorrect {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* 학습 플랜 카드 */
    .plan-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: black;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
    }
    
    /* 컨디션 상태 */
    .condition-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* 임베드 컨테이너 */
    .embed-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* 진행률 바 */
    .progress-container {
        background: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)

st.markdown("""
이 앱은 인도네시아어 초급 학습자를 위한 도구입니다.  
**오디오 파일**, **YouTube 링크**, **텍스트 웹 링크**를 입력하면 교육적 가치를 분석하고 퀴즈를 생성합니다.
""")

# =====================================================
# 사이드바 설정
# =====================================================
logo_label = "🎓 언어학습앱"
st.markdown(f"""
<style>
div[data-testid="stSidebar"] button[aria-label="{logo_label}"] {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;

  padding: 0 !important;
  margin: 10px 0 18px 0 !important;

  width: auto !important;
  min-height: 0 !important;

  border-radius: 0 !important;
}}

div[data-testid="stSidebar"] button[aria-label="{logo_label}"] p {{
  margin: 0 !important;
  font-size: 22px !important;
  font-weight: 800 !important;
  line-height: 1.1 !important;
}}

div[data-testid="stSidebar"] button[aria-label="{logo_label}"]:hover,
div[data-testid="stSidebar"] button[aria-label="{logo_label}"]:active,
div[data-testid="stSidebar"] button[aria-label="{logo_label}"]:focus,
div[data-testid="stSidebar"] button[aria-label="{logo_label}"]:focus-visible {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}}

/* 보험: 사이드바 첫 버튼을 로고로 간주 */
div[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type button {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  width: auto !important;
  border-radius: 0 !important;
}}
</style>
""", unsafe_allow_html=True)

# 로고 클릭 카운터 초기화
if "logo_click_count" not in st.session_state:
    st.session_state["logo_click_count"] = 0
if "last_logo_click_time" not in st.session_state:
    st.session_state["last_logo_click_time"] = 0
if "debug_mode_enabled" not in st.session_state:
    st.session_state["debug_mode_enabled"] = False

with st.sidebar:
    # 로고 버튼 (히든 디버그 토글)
    if st.button(logo_label, key="logo_button", type="secondary"):
        current_time = time.time()

        # 3초 이내에 클릭하면 카운터 증가, 아니면 리셋
        if current_time - st.session_state["last_logo_click_time"] < 3:
            st.session_state["logo_click_count"] += 1
        else:
            st.session_state["logo_click_count"] = 1

        st.session_state["last_logo_click_time"] = current_time

        # 5번 클릭하면 디버그 모드 토글
        if st.session_state["logo_click_count"] >= 5:
            st.session_state["debug_mode_enabled"] = not st.session_state["debug_mode_enabled"]
            st.session_state["logo_click_count"] = 0

            st.toast(
                "🔍 디버그 모드 활성화!" if st.session_state["debug_mode_enabled"] else "🔒 디버그 모드 비활성화",
                icon="🔓" if st.session_state["debug_mode_enabled"] else "🔒"
            )
            st.rerun()

    # 이하 학습 설정/모델 설정 코드 계속...

    
    # 학습 설정
    st.subheader("📚 학습 설정")
    
    condition = st.selectbox(
        "오늘 컨디션", 
        ["A (여유)", "B (보통)", "C (힘듦)"], 
        index=None,  # 기본 선택 없음
        placeholder="컨디션을 선택하세요",
        help="컨디션에 따라 문제 수가 달라집니다 (A: 10문제, B: 5문제, C: 3문제)"
    )
    
    # 컨디션에 따른 문제 수 매핑
    if condition:
        condition_to_questions = {
            "A": 10,
            "B": 5,
            "C": 3
        }
        condition_simple = condition.split()[0]
        num_questions = condition_to_questions.get(condition_simple, 5)
        st.caption(f"💡 현재 설정: **{num_questions}문제** 생성")
    else:
        num_questions = 5  # 기본값
        st.caption("⚠️ 컨디션을 선택하지 않았습니다 (기본: 5문제)")
    
    mode = st.selectbox(
        "학습 모드", 
        ["BIPA (초급)", "BIPA (중급)"], 
        index=None,  # 기본 선택 없음
        placeholder="학습 모드를 선택하세요"
    )
    
    # 학습 모드에 따른 레벨 매핑
    if mode:
        mode_to_level = {
            "BIPA (초급)": "초급 (A1~A2)",
            "BIPA (중급)": "중급 (B1~B2)"
        }
        level = mode_to_level.get(mode, "초급 (A1~A2)")
    else:
        level = "초급 (A1~A2)"  # 기본값
        st.caption("⚠️ 학습 모드를 선택하지 않았습니다 (기본: 초급)")
    
    st.divider()
    
    # 모델 설정
    st.subheader("🤖 모델 설정")
    gen_model = st.text_input("생성 모델", value="gpt-4o-mini")
    
    # 디버그 모드 표시 (활성화된 경우)
    if st.session_state.get("debug_mode_enabled", False):
        st.success("🔍 DEBUG 모드 활성화됨")
        if st.button("❌ 디버그 모드 비활성화", key="disable_debug"):
            st.session_state["debug_mode_enabled"] = False
            st.rerun()

# 디버그 모드 변수 (전역에서 사용 가능하도록)
debug = st.session_state.get("debug_mode_enabled", False)

# =====================================================
# 탭 구성
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎵 오디오 학습",
    "📺 YouTube 학습",
    "📄 텍스트 학습",
    "📊 학습 결과",
    "⚙️ 설정"
])

# =====================================================
# TAB 1: 오디오 학습
# =====================================================

with tab1:
    st.header("🎵 오디오로 학습하기")
    st.markdown("WAV 파일을 업로드하면 음성을 텍스트로 변환하고 퀴즈를 생성합니다.")
    
    # 1단계: 오디오 선택
    st.subheader("1️⃣ 오디오 선택")
    
    col_audio1, col_audio2 = st.columns([3, 1])
    
    with col_audio1:
        use_sample = st.checkbox("샘플 오디오 사용", value=True, key="use_sample_audio")
    
    # uploaded 변수 초기화 (변수 정의 오류 방지)
    uploaded = None
    
    with col_audio2:
        if not use_sample:
            uploaded = st.file_uploader("WAV 업로드", type=["wav"], key="audio_uploader", label_visibility="collapsed")
    
    wav_path = None
    if use_sample:
        # 샘플 파일 경로 수정 (프로젝트 루트에 위치)
        sample_paths = [f for f in ["sample_A.wav", "sample_B.wav"] if os.path.exists(f)]
        if sample_paths:
            sample_choice = st.radio("샘플 선택", sample_paths, index=0, horizontal=True)
            wav_path = sample_choice
        else:
            # 샘플 파일이 없을 경우 안내 메시지
            st.warning("⚠️ 샘플 오디오 파일(sample_A.wav, sample_B.wav)을 찾을 수 없습니다. 파일을 업로드해주세요.")
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
        if st.button("🎤 음성 → 텍스트 변환", type="primary", key="btn_asr", use_container_width=True):
            asr = load_asr()
            t0 = time.perf_counter()
            
            try:
                with st.spinner("음성을 텍스트로 변환 중... (CPU에서는 시간이 걸릴 수 있습니다)"):
                    transcript = transcribe_audio(asr, wav_path)
                    # 가독성을 위해 오디오 전용 포맷팅 적용 (문장 단위로 3개씩 문단 구분)
                    formatted_transcript = format_audio_transcript(transcript, sentences_per_paragraph=3)
                    st.session_state["audio_transcript"] = formatted_transcript
                    st.session_state["current_source"] = f"Audio: {os.path.basename(wav_path)}"
                    # 퀴즈 초기화
                    st.session_state.pop("audio_quiz", None)
                    st.session_state.pop("audio_coach", None)
                
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
        
        st.text_area("인도네시아어 텍스트", value=audio_transcript, height=500, key="audio_transcript_display", disabled=True)
        # 문단 수 계산
        paragraph_count = audio_transcript.count("\n\n") + 1
        st.caption(f"📊 텍스트 길이: {len(audio_transcript)}자 | 문단 수: {paragraph_count}개")
        
        # 퀴즈 생성 버튼
        st.markdown("---")
        
        if st.button(f"🎯 퀴즈 {num_questions}문항 생성하기", type="primary", key="btn_generate_audio_quiz_main", use_container_width=True):
            st.session_state["start_audio_quiz_generation"] = True
            st.rerun()
    
    # 퀴즈 생성 처리
    if st.session_state.get("start_audio_quiz_generation"):
        st.divider()
        st.subheader("3️⃣ 퀴즈 생성 중...")
        
        if audio_transcript:
            try:
                quiz_text = audio_transcript[:4000] if len(audio_transcript) > 4000 else audio_transcript
                prompt = QUIZ_PROMPT.format(
                    transcript=quiz_text,
                    num_questions=num_questions,
                    level=level
                )
                
                if debug:
                    with st.expander("🔍 DEBUG: QUIZ_PROMPT"):
                        st.code(prompt[:1000])
                
                with st.spinner("퀴즈를 생성 중... (약 10초 소요)"):
                    quiz = llm_json(prompt, model=gen_model)
                
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
                            f"답 선택",
                            options=opts,
                            format_func=lambda k, choices=choices: f"{k}. {choices.get(k, '')}",
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
                    empty_answers = [qid for qid, ans in user_answers.items() if not ans]
                    if empty_answers:
                        st.error(f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})")
                    else:
                        try:
                            condition_simple = condition.split()[0] if condition else "B"
                            
                            prompt = COACH_PROMPT.format(
                                transcript=audio_transcript[:4000],
                                quiz_json=json.dumps(audio_quiz, ensure_ascii=False),
                                user_answers=json.dumps(user_answers, ensure_ascii=False),
                                condition=condition_simple,
                            )
                            
                            with st.spinner("채점 중... (Structured Outputs 사용)"):
                                # Structured Outputs 사용
                                coach = llm_structured(prompt, CoachResponse, model=gen_model)
                                coach = sanitize_coach_structured(coach, audio_quiz, user_answers)
                            
                            st.session_state["audio_coach"] = coach
                            
                            # 학습 기록 저장
                            wrong_items_analyzed = []
                            for item in coach.get("wrong_items", []):
                                q = next((q for q in audio_quiz.get("questions", []) if str(q.get("id")) == str(item.get("id"))), {})
                                analyzed = WeaknessAnalyzer.analyze_wrong_answer(q, item.get("user_answer", ""), item.get("correct_answer", ""))
                                analyzed.update(item)
                                wrong_items_analyzed.append(analyzed)
                            
                            LearningHistoryManager.add_session({
                                "source": "audio",
                                "level": level,
                                "condition": condition,
                                "score": coach.get("score", {}),
                                "wrong_items": wrong_items_analyzed,
                            })
                            
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
            col3.metric("컨디션", condition.split()[0] if condition else "미설정")
            
            # 오답 풀이 및 해설
            st.divider()
            st.markdown("#### 📝 문제 풀이 및 해설")
            
            audio_quiz = st.session_state.get("audio_quiz", {})
            questions = audio_quiz.get("questions", [])
            wrong_items = audio_coach.get("wrong_items", [])
            wrong_ids = [str(wi.get("id")) for wi in wrong_items]
            
            for q in questions:
                qid = str(q.get("id"))
                is_wrong = qid in wrong_ids
                
                # 정답/오답 표시
                if is_wrong:
                    st.markdown(f"**Q{qid}. {q.get('question', '')}** ❌")
                else:
                    st.markdown(f"**Q{qid}. {q.get('question', '')}** ✅")
                
                choices = q.get("choices", {})
                correct_ans = q.get("answer", "")
                
                # 오답인 경우 상세 해설 표시
                if is_wrong:
                    wrong_item = next((wi for wi in wrong_items if str(wi.get("id")) == qid), None)
                    if wrong_item:
                        user_ans = wrong_item.get("user_answer", "")
                        
                        # 내 답 vs 정답
                        st.warning(f"**내 답:** {user_ans} | **정답:** {correct_ans}")
                        
                        # 정답이 정답인 이유
                        why_correct = wrong_item.get("why_correct_ko", "")
                        if why_correct:
                            st.success(f"✅ **정답 해설:** {why_correct}")
                        
                        # 내 답이 틀린 이유
                        why_user_wrong = wrong_item.get("why_user_wrong_ko", "")
                        if why_user_wrong:
                            st.error(f"❌ **오답 이유:** {why_user_wrong}")
                        
                        # 근거 인용
                        evidence = wrong_item.get("evidence_quote", "")
                        if evidence:
                            st.info(f"📄 **원문 근거:** \"{evidence}\"")
                        
                        # 각 보기 해설
                        choices_exp = wrong_item.get("choices_explanation", {})
                        if choices_exp:
                            st.markdown("**📋 보기별 해설:**")
                            for opt in ["A", "B", "C", "D"]:
                                exp = choices_exp.get(opt, "")
                                choice_text = choices.get(opt, "")
                                if opt == correct_ans:
                                    st.markdown(f"- **{opt}. {choice_text}** ✓ → {exp}")
                                else:
                                    st.markdown(f"- {opt}. {choice_text} → {exp}")
                else:
                    # 정답인 경우 선택지만 표시
                    for opt in ["A", "B", "C", "D"]:
                        choice_text = choices.get(opt, "")
                        if opt == correct_ans:
                            st.markdown(f"- **{opt}. {choice_text}** ✓ (정답)")
                        else:
                            st.markdown(f"- {opt}. {choice_text}")
                
                st.markdown("")  # 여백
            
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

# =====================================================
# TAB 2: YouTube 학습
# =====================================================

with tab2:
    st.header("📺 YouTube로 학습하기")
    st.markdown("YouTube 영상을 시청하고 인도네시아어 요약을 작성한 후 퀴즈를 풀어보세요!")
    
    st.warning("""
    ⚠️ **중요 사항**:
    - YouTube 영상은 임베드 형태로만 제공됩니다.
    - 자동으로 자막이나 오디오를 다운로드하지 않습니다.
    - 사용자가 직접 시청하고 메모한 내용을 입력해주세요.
    """)
    
    # YouTube URL 입력
    st.subheader("1️⃣ YouTube 영상 선택")
    
    col_url1, col_url2 = st.columns([3, 1])
    
    # 샘플 로드 플래그 확인 (이전 rerun에서 설정된 경우)
    if st.session_state.get("load_sample_flag"):
        st.session_state["youtube_url_input"] = "https://www.youtube.com/watch?v=WOt9_kqiZtw"
        st.session_state.pop("load_sample_flag")  # 플래그 제거
        reset_learning_state("youtube")
        st.session_state.pop("prev_youtube_video_id", None)
    
    with col_url1:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
            help="YouTube 영상 URL을 입력하세요"
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
        
        if prev_video_id and prev_video_id != current_video_id and not fetching_transcript:
            # video_id가 변경되었으면 이전 데이터가 있는지 확인
            had_data = (
                st.session_state.get("youtube_quiz") is not None or 
                st.session_state.get("youtube_coach") is not None or
                st.session_state.get("youtube_transcript") is not None
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
            st.markdown(f"""
            <iframe width="100%" height="400" 
            src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
            </iframe>
            """, unsafe_allow_html=True)
            
            st.caption(f"출처: YouTube | {youtube_url}")
            st.session_state["current_source"] = f"YouTube: {youtube_url}"
            
            # 현재 영상 정보 표시
            st.info(f"📹 **현재 영상 ID:** `{video_id}`")
            
            # 기존 퀴즈가 다른 영상의 것인지 확인
            saved_video_id = st.session_state.get("youtube_quiz_video_id", "")
            if saved_video_id and saved_video_id != video_id:
                st.warning(f"⚠️ 이전 퀴즈는 다른 영상(`{saved_video_id}`)의 것입니다. 새로운 요약을 작성하고 퀴즈를 다시 생성해주세요.")
            
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
                fetch_clicked = st.button("🎬 자막 가져오기", key=f"fetch_subtitle_{video_id}", use_container_width=True)
            
            with col_sub2:
                reset_subtitle_clicked = st.button("🔄 자막 초기화", key=f"reset_subtitle_{video_id}")
            
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
                        st.success(f"✅ 인도네시아어 자막을 성공적으로 가져왔습니다! ({len(result['transcript'])}자)")
                    else:
                        st.warning(f"⚠️ 인도네시아어 자막이 없어 {lang_used} 자막을 가져왔습니다. ({len(result['transcript'])}자)")
                        st.info("💡 아래에서 자막을 확인하고 참고하여 요약을 작성하세요.")
                else:
                    st.error(f"❌ {result['error']}")
                    st.info("💡 자막이 없는 경우 직접 영상을 시청하고 아래에 요약을 작성해주세요.")
            
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
                    label_visibility="collapsed"
                )
                # 문단 수 계산 (빈 줄 기준)
                paragraph_count = fetched_subtitle.count("\n\n") + 1
                st.caption(f"📊 자막 길이: {len(fetched_subtitle)}자 | 문단 수: {paragraph_count}개")
            
            # ===== 3️⃣ 영상 내용 요약 작성 섹션 =====
            st.divider()
            st.subheader("3️⃣ 영상 내용 요약 작성")
            
            st.markdown(f"""
            **📝 이 영상(`{video_id}`)에 대한 요약을 작성해주세요.**
            
            - 위에서 자막을 가져왔다면 참고하여 요약을 작성하세요.
            - 또는 영상을 시청하고 직접 내용을 정리하세요.
            """)
            
            with st.expander("💡 인도네시아어 요약 작성 팁", expanded=False):
                st.markdown("""
                **좋은 요약을 작성하는 방법:**
                
                1. **주요 내용 3-5가지**를 인도네시아어로 작성
                2. **완전한 문장**으로 작성 (주어 + 동사 + 목적어)
                3. **구체적인 정보** 포함 (숫자, 이름, 장소 등)
                4. **최소 5문장** 이상 작성
                
                **예시:**
                ```
                Video ini membahas tentang sistem pendidikan di Amerika Serikat.
                Guru Indonesia menjelaskan perbedaan antara sekolah di Indonesia dan Amerika.
                Di Amerika, siswa dapat memilih mata pelajaran yang mereka sukai.
                Sistem pendidikan di Amerika lebih fleksibel dibandingkan Indonesia.
                Banyak sekolah di Amerika memiliki fasilitas yang sangat baik.
                ```
                """)
            
            # 요약 입력 (사용자가 직접 작성)
            summary_key = f"youtube_user_summary_{video_id}"
            youtube_summary_input = st.text_area(
                "📝 영상 내용 요약 (인도네시아어)",
                height=250,
                placeholder="""영상을 시청한 후, 들은 내용을 인도네시아어로 요약하세요.

예시:
Video ini membahas tentang...
Pembicara menjelaskan bahwa...
Topik utama adalah...""",
                key=summary_key,
                help="최소 50자 이상 작성하세요"
            )
            
            # 글자 수 표시
            char_count = len(youtube_summary_input.strip()) if youtube_summary_input else 0
            
            if char_count > 0:
                if char_count >= 50:
                    st.success(f"✅ 작성 완료: {char_count}자 (최소 50자)")
                else:
                    st.warning(f"⚠️ {char_count}자 / 최소 50자 필요 (아직 {50 - char_count}자 더 필요)")
            
            # 퀴즈 생성 버튼 - 요약 작성 바로 아래
            st.markdown("---")
            
            # 사용자 요약이 있으면 우선 사용, 없으면 자막 사용
            text_for_quiz = youtube_summary_input.strip() if youtube_summary_input.strip() else fetched_subtitle
            quiz_char_count = len(text_for_quiz)
            
            if quiz_char_count >= 50:
                # 버튼 key도 video_id를 포함시켜 URL마다 독립적으로
                quiz_btn_key = f"btn_generate_youtube_quiz_{video_id}"
                
                # 어떤 자료로 퀴즈를 생성하는지 표시
                if youtube_summary_input.strip():
                    btn_label = f"🎯 작성한 요약으로 퀴즈 {num_questions}문항 생성"
                else:
                    btn_label = f"🎯 가져온 자막으로 퀴즈 {num_questions}문항 생성"
                
                if st.button(btn_label, type="primary", key=quiz_btn_key, use_container_width=True):
                    # session_state에 플래그 및 현재 URL 정보 저장
                    st.session_state["start_quiz_generation"] = True
                    st.session_state["youtube_transcript"] = text_for_quiz
                    st.session_state["youtube_quiz_video_id"] = video_id  # 현재 비디오 ID 저장
                    st.session_state["youtube_current_url"] = youtube_url  # 현재 URL도 저장
                    # 이전 퀴즈 강제 초기화 (새 영상의 퀴즈 생성 보장)
                    st.session_state.pop("youtube_quiz", None)
                    st.session_state.pop("youtube_coach", None)
                    st.rerun()
            else:
                quiz_btn_disabled_key = f"btn_generate_youtube_quiz_disabled_{video_id}"
                st.button(f"🎯 퀴즈 {num_questions}문항 생성하기", type="primary", key=quiz_btn_disabled_key, use_container_width=True, disabled=True)
                if fetched_subtitle:
                    st.caption("💡 자막을 가져왔으므로 바로 퀴즈를 생성할 수 있습니다. 또는 요약을 작성하세요.")
                else:
                    st.caption("💡 자막을 가져오거나 요약을 작성하면 (최소 50자) 버튼이 활성화됩니다.")
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
                quiz_text = saved_transcript[:4000] if len(saved_transcript) > 4000 else saved_transcript
                prompt = QUIZ_PROMPT.format(
                    transcript=quiz_text,
                    num_questions=num_questions,
                    level=level
                )
                
                if debug:
                    with st.expander("🔍 DEBUG: QUIZ_PROMPT"):
                        st.code(prompt[:1000])
                
                with st.spinner(f"영상 `{generating_video_id}`에 대한 퀴즈를 생성 중... (약 10초 소요)"):
                    quiz = llm_json(prompt, model=gen_model)
                
                st.session_state["youtube_quiz"] = quiz
                st.session_state.pop("youtube_coach", None)
                st.session_state.pop("start_quiz_generation")  # 플래그 제거
                st.success(f"✅ 퀴즈 생성 완료! (영상 ID: `{generating_video_id}`)")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ 퀴즈 생성 실패 (영상 ID: `{generating_video_id}`)")
                st.exception(e)
                st.session_state.pop("start_quiz_generation", None)  # 오류 시에도 플래그 제거
    
    # 퀴즈 표시 및 답안 입력
    youtube_quiz = st.session_state.get("youtube_quiz")
    
    # 현재 URL의 퀴즈인지 확인 (video_id 일치 여부)
    if youtube_quiz and youtube_url:
        current_video_id = extract_youtube_id(youtube_url)
        saved_video_id = st.session_state.get("youtube_quiz_video_id", "")
        
        # 비디오 ID가 다르면 퀴즈 무효화
        if current_video_id != saved_video_id:
            st.warning(f"⚠️ 표시된 퀴즈는 다른 영상(`{saved_video_id}`)의 것입니다. 현재 영상(`{current_video_id}`)에 대한 퀴즈를 생성하려면 위에서 요약을 작성하고 퀴즈 생성 버튼을 눌러주세요.")
            youtube_quiz = None
            st.session_state.pop("youtube_quiz", None)
            st.session_state.pop("youtube_coach", None)
    
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
                            f"답 선택",
                            options=opts,
                            format_func=lambda k, choices=choices: f"{k}. {choices.get(k, '')}",
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
                    empty_answers = [qid for qid, ans in user_answers.items() if not ans]
                    if empty_answers:
                        st.error(f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})")
                    else:
                        try:
                            condition_simple = condition.split()[0] if condition else "B"
                            saved_transcript = st.session_state.get("youtube_transcript", "")
                            
                            prompt = COACH_PROMPT.format(
                                transcript=saved_transcript[:4000],
                                quiz_json=json.dumps(youtube_quiz, ensure_ascii=False),
                                user_answers=json.dumps(user_answers, ensure_ascii=False),
                                condition=condition_simple,
                            )
                            
                            with st.spinner("채점 중... (Structured Outputs 사용)"):
                                # Structured Outputs 사용
                                coach = llm_structured(prompt, CoachResponse, model=gen_model)
                                coach = sanitize_coach_structured(coach, youtube_quiz, user_answers)
                            
                            st.session_state["youtube_coach"] = coach
                            
                            # 학습 기록 저장
                            wrong_items_analyzed = []
                            for item in coach.get("wrong_items", []):
                                q = next((q for q in youtube_quiz.get("questions", []) if str(q.get("id")) == str(item.get("id"))), {})
                                analyzed = WeaknessAnalyzer.analyze_wrong_answer(q, item.get("user_answer", ""), item.get("correct_answer", ""))
                                analyzed.update(item)
                                wrong_items_analyzed.append(analyzed)
                            
                            LearningHistoryManager.add_session({
                                "source": "youtube",
                                "level": level,
                                "condition": condition,
                                "score": coach.get("score", {}),
                                "wrong_items": wrong_items_analyzed,
                            })
                            
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
            col3.metric("컨디션", condition.split()[0] if condition else "미설정")
            col4.metric("영상 ID", f"{result_video_id[:8]}...")
            
            # 오답 풀이 및 해설
            st.divider()
            st.markdown("#### 📝 문제 풀이 및 해설")
            
            youtube_quiz = st.session_state.get("youtube_quiz", {})
            questions = youtube_quiz.get("questions", [])
            wrong_items = youtube_coach.get("wrong_items", [])
            wrong_ids = [str(wi.get("id")) for wi in wrong_items]
            
            for q in questions:
                qid = str(q.get("id"))
                is_wrong = qid in wrong_ids
                
                # 정답/오답 표시
                if is_wrong:
                    st.markdown(f"**Q{qid}. {q.get('question', '')}** ❌")
                else:
                    st.markdown(f"**Q{qid}. {q.get('question', '')}** ✅")
                
                choices = q.get("choices", {})
                correct_ans = q.get("answer", "")
                
                # 오답인 경우 상세 해설 표시
                if is_wrong:
                    wrong_item = next((wi for wi in wrong_items if str(wi.get("id")) == qid), None)
                    if wrong_item:
                        user_ans = wrong_item.get("user_answer", "")
                        
                        # 내 답 vs 정답
                        st.warning(f"**내 답:** {user_ans} | **정답:** {correct_ans}")
                        
                        # 정답이 정답인 이유
                        why_correct = wrong_item.get("why_correct_ko", "")
                        if why_correct:
                            st.success(f"✅ **정답 해설:** {why_correct}")
                        
                        # 내 답이 틀린 이유
                        why_user_wrong = wrong_item.get("why_user_wrong_ko", "")
                        if why_user_wrong:
                            st.error(f"❌ **오답 이유:** {why_user_wrong}")
                        
                        # 근거 인용
                        evidence = wrong_item.get("evidence_quote", "")
                        if evidence:
                            st.info(f"📄 **원문 근거:** \"{evidence}\"")
                        
                        # 각 보기 해설
                        choices_exp = wrong_item.get("choices_explanation", {})
                        if choices_exp:
                            st.markdown("**📋 보기별 해설:**")
                            for opt in ["A", "B", "C", "D"]:
                                exp = choices_exp.get(opt, "")
                                choice_text = choices.get(opt, "")
                                if opt == correct_ans:
                                    st.markdown(f"- **{opt}. {choice_text}** ✓ → {exp}")
                                else:
                                    st.markdown(f"- {opt}. {choice_text} → {exp}")
                else:
                    # 정답인 경우 선택지만 표시
                    for opt in ["A", "B", "C", "D"]:
                        choice_text = choices.get(opt, "")
                        if opt == correct_ans:
                            st.markdown(f"- **{opt}. {choice_text}** ✓ (정답)")
                        else:
                            st.markdown(f"- {opt}. {choice_text}")
                
                st.markdown("")  # 여백
            
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
    else:
        st.info("""
        📝 **요약을 작성해주세요!**
        
        1. 위에서 YouTube 영상을 시청하세요
        2. 영상 내용을 인도네시아어로 요약하세요 (최소 50자)
        3. 퀴즈 생성 버튼이 나타납니다
        
        💡 **팁**: 최소 5문장 이상 작성하면 좋은 퀴즈가 생성됩니다!
        """)

# =====================================================
# TAB 3: 텍스트 학습
# =====================================================

with tab3:
    st.header("📄 텍스트로 학습하기")
    st.markdown("웹 링크를 입력하면 텍스트를 추출하여 학습 자료로 사용합니다.")
    
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
            help="VOA Indonesia, Wikisource 등의 URL을 입력하세요"
        )
    
    with col_url2:
        st.markdown("&nbsp;")  # 공백
        extract_btn = st.button("🔍 추출", key="btn_extract_text", type="primary", use_container_width=True)
    
    # URL 입력 안내
    if text_url and not extract_btn:
        st.info("👉 URL을 입력했습니다. 위의 **'🔍 추출'** 버튼을 눌러 텍스트를 가져오세요.")
    
    # 추출 버튼 처리
    if extract_btn:
        if text_url:
            with st.spinner(f"'{text_url}'에서 텍스트를 추출 중..."):
                result = extract_text_from_url(text_url)
            
            if result["success"]:
                st.session_state["extracted_text"] = result["text"]
                st.session_state["extracted_title"] = result["title"]
                st.session_state["current_source"] = f"Web: {text_url}"
                st.session_state["current_text_url"] = text_url  # 현재 URL 저장
                st.session_state.pop("text_quiz", None)
                st.session_state.pop("text_coach", None)
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
            disabled=True
        )
        
        # 문단 수 계산
        paragraph_count = extracted_text.count("\n\n") + 1
        st.caption(f"📊 텍스트 길이: {len(extracted_text)}자 | 문단 수: {paragraph_count}개")
        
        # 퀴즈 생성 버튼
        st.markdown("---")
        
        if st.button(f"🎯 퀴즈 {num_questions}문항 생성하기", type="primary", key="btn_generate_text_quiz_main", use_container_width=True):
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
                prompt = QUIZ_PROMPT.format(
                    transcript=quiz_text,
                    num_questions=num_questions,
                    level=level
                )
                
                if debug:
                    with st.expander("🔍 DEBUG: QUIZ_PROMPT"):
                        st.code(prompt[:1000])
                
                with st.spinner("퀴즈를 생성 중... (약 10초 소요)"):
                    quiz = llm_json(prompt, model=gen_model)
                
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
                            f"답 선택",
                            options=opts,
                            format_func=lambda k, choices=choices: f"{k}. {choices.get(k, '')}",
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
                    empty_answers = [qid for qid, ans in user_answers.items() if not ans]
                    if empty_answers:
                        st.error(f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})")
                    else:
                        try:
                            condition_simple = condition.split()[0] if condition else "B"
                            
                            prompt = COACH_PROMPT.format(
                                transcript=extracted_text[:4000],
                                quiz_json=json.dumps(text_quiz, ensure_ascii=False),
                                user_answers=json.dumps(user_answers, ensure_ascii=False),
                                condition=condition_simple,
                            )
                            
                            with st.spinner("채점 중... (Structured Outputs 사용)"):
                                # Structured Outputs 사용
                                coach = llm_structured(prompt, CoachResponse, model=gen_model)
                                coach = sanitize_coach_structured(coach, text_quiz, user_answers)
                            
                            st.session_state["text_coach"] = coach
                            
                            # 학습 기록 저장
                            wrong_items_analyzed = []
                            for item in coach.get("wrong_items", []):
                                q = next((q for q in text_quiz.get("questions", []) if str(q.get("id")) == str(item.get("id"))), {})
                                analyzed = WeaknessAnalyzer.analyze_wrong_answer(q, item.get("user_answer", ""), item.get("correct_answer", ""))
                                analyzed.update(item)
                                wrong_items_analyzed.append(analyzed)
                            
                            LearningHistoryManager.add_session({
                                "source": "text",
                                "level": level,
                                "condition": condition,
                                "score": coach.get("score", {}),
                                "wrong_items": wrong_items_analyzed,
                            })
                            
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
            col3.metric("컨디션", condition.split()[0] if condition else "미설정")
            
            # 오답 풀이 및 해설
            st.divider()
            st.markdown("#### 📝 문제 풀이 및 해설")
            
            text_quiz = st.session_state.get("text_quiz", {})
            questions = text_quiz.get("questions", [])
            wrong_items = text_coach.get("wrong_items", [])
            wrong_ids = [str(wi.get("id")) for wi in wrong_items]
            
            for q in questions:
                qid = str(q.get("id"))
                is_wrong = qid in wrong_ids
                
                # 정답/오답 표시
                if is_wrong:
                    st.markdown(f"**Q{qid}. {q.get('question', '')}** ❌")
                else:
                    st.markdown(f"**Q{qid}. {q.get('question', '')}** ✅")
                
                choices = q.get("choices", {})
                correct_ans = q.get("answer", "")
                
                # 오답인 경우 상세 해설 표시
                if is_wrong:
                    wrong_item = next((wi for wi in wrong_items if str(wi.get("id")) == qid), None)
                    if wrong_item:
                        user_ans = wrong_item.get("user_answer", "")
                        
                        # 내 답 vs 정답
                        st.warning(f"**내 답:** {user_ans} | **정답:** {correct_ans}")
                        
                        # 정답이 정답인 이유
                        why_correct = wrong_item.get("why_correct_ko", "")
                        if why_correct:
                            st.success(f"✅ **정답 해설:** {why_correct}")
                        
                        # 내 답이 틀린 이유
                        why_user_wrong = wrong_item.get("why_user_wrong_ko", "")
                        if why_user_wrong:
                            st.error(f"❌ **오답 이유:** {why_user_wrong}")
                        
                        # 근거 인용
                        evidence = wrong_item.get("evidence_quote", "")
                        if evidence:
                            st.info(f"📄 **원문 근거:** \"{evidence}\"")
                        
                        # 각 보기 해설
                        choices_exp = wrong_item.get("choices_explanation", {})
                        if choices_exp:
                            st.markdown("**📋 보기별 해설:**")
                            for opt in ["A", "B", "C", "D"]:
                                exp = choices_exp.get(opt, "")
                                choice_text = choices.get(opt, "")
                                if opt == correct_ans:
                                    st.markdown(f"- **{opt}. {choice_text}** ✓ → {exp}")
                                else:
                                    st.markdown(f"- {opt}. {choice_text} → {exp}")
                else:
                    # 정답인 경우 선택지만 표시
                    for opt in ["A", "B", "C", "D"]:
                        choice_text = choices.get(opt, "")
                        if opt == correct_ans:
                            st.markdown(f"- **{opt}. {choice_text}** ✓ (정답)")
                        else:
                            st.markdown(f"- {opt}. {choice_text}")
                
                st.markdown("")  # 여백
            
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

# =====================================================
# TAB 4: 학습 결과
# =====================================================

with tab4:
    st.header("📊 학습 결과 및 분석")
    
    # 탭 구성 확장: 대시보드 | 반복 학습 | 섀도잉 | 현재 세션 | SRS 복습
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "📈 대시보드",
        "🔄 반복 학습",
        "🗣️ 섀도잉",
        "📝 현재 세션",
        "📅 SRS 복습"
    ])
    
    # ==========================================
    # 서브탭 1: 학습 대시보드
    # ==========================================
    with subtab1:
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
                delta=f"+{history_stats['sessions_this_week']}회 (이번 주)"
            )
        
        with col2:
            delta_color = "normal" if history_stats['score_trend'] >= 0 else "inverse"
            st.metric(
                "평균 정답률",
                f"{history_stats['avg_score']}%",
                delta=f"{history_stats['score_trend']:+d}% (추세)",
                delta_color=delta_color
            )
        
        with col3:
            st.metric(
                "연속 학습",
                f"{history_stats['streak_days']}일",
                delta="🔥 유지 중!" if history_stats['streak_days'] > 0 else None
            )
        
        with col4:
            st.metric(
                "오늘 복습 예정",
                f"{srs_stats['due_today']}개",
                delta=f"전체 {srs_stats['total_items']}개 중"
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
                
                bar = "█" * sessions + "░" * (max(5, max(d["sessions"] for d in daily_stats)) - sessions)
                score_bar = "●" * (score // 10) + "○" * (10 - score // 10) if score > 0 else "—" * 10
                
                col_date, col_bar, col_score = st.columns([1, 2, 2])
                with col_date:
                    st.caption(date_label)
                with col_bar:
                    st.caption(f"세션: {bar} ({sessions})")
                with col_score:
                    st.caption(f"점수: {score_bar} ({score}%)" if score > 0 else "점수: — (없음)")
        else:
            st.info("아직 학습 기록이 없습니다. 퀴즈를 풀어보세요!")
        
        st.divider()
        
        # 취약점 분석
        st.markdown("#### 🎯 취약 카테고리 분석")
        
        weakness = LearningHistoryManager.get_weakness_analysis(10)
        
        if weakness["total_wrong"] > 0:
            st.caption(f"최근 10개 세션 기준, 총 {weakness['total_wrong']}개 오답 분석")
            
            for rec in weakness.get("recommendations", [])[:3]:
                cat_icon = rec.get("icon", "📌")
                cat_name = rec.get("name", rec.get("category", ""))
                count = rec.get("count", 0)
                activity = rec.get("activity", "")
                
                progress = count / weakness["total_wrong"] if weakness["total_wrong"] > 0 else 0
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                    <span style="font-size: 1.5rem;">{cat_icon}</span>
                    <strong>{cat_name}</strong>: {count}개 오답 ({progress*100:.0f}%)
                    <br><small style="color: #666;">👉 {activity}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 최근 오답이 없습니다! 훌륭해요!")
        
        st.divider()
        
        # SRS 카테고리별 통계
        st.markdown("#### 📚 카테고리별 학습 현황")
        
        cat_stats = SpacedRepetitionSystem.get_category_stats()
        
        if cat_stats:
            for cat_key, stats in cat_stats.items():
                cat_info = CEFR_CATEGORIES.get(cat_key, {"name": cat_key, "icon": "📌"})
                total = stats["total"]
                mastered = stats["mastered"]
                accuracy = int((stats["correct"] / stats["reviews"]) * 100) if stats["reviews"] > 0 else 0
                
                col_cat, col_progress, col_accuracy = st.columns([2, 3, 1])
                with col_cat:
                    st.markdown(f"{cat_info['icon']} **{cat_info['name']}**")
                with col_progress:
                    st.progress(mastered / total if total > 0 else 0, text=f"마스터: {mastered}/{total}")
                with col_accuracy:
                    st.caption(f"정확도: {accuracy}%")
        else:
            st.info("SRS에 등록된 항목이 없습니다.")
    
    # ==========================================
    # 서브탭 2: 반복 학습 (틀린 문제 정답까지)
    # ==========================================
    with subtab2:
        st.subheader("🔄 틀린 문제 반복 학습")
        st.info("💡 틀린 문제를 모두 맞출 때까지 반복합니다. 유사 문제로 추가 연습도 가능합니다.")
        
        # 현재 진행 상황 확인
        progress = RepeatLearningManager.get_progress()
        
        # 시작되지 않은 경우 - 틀린 문제 불러오기
        if not progress["active"]:
            st.markdown("#### 📋 반복 학습 시작하기")
            
            # 현재 코칭 결과에서 오답 확인
            any_coach = (
                st.session_state.get("audio_coach") or 
                st.session_state.get("youtube_coach") or 
                st.session_state.get("text_coach")
            )
            any_quiz = (
                st.session_state.get("audio_quiz") or 
                st.session_state.get("youtube_quiz") or 
                st.session_state.get("text_quiz")
            )
            
            if any_coach and any_coach.get("wrong_items"):
                wrong_items = any_coach.get("wrong_items", [])
                quiz_questions = any_quiz.get("questions", []) if any_quiz else []
                
                st.success(f"✅ {len(wrong_items)}개의 틀린 문제가 있습니다.")
                
                # 틀린 문제 미리보기
                with st.expander("🔍 틀린 문제 미리보기", expanded=False):
                    for item in wrong_items:
                        q_id = item.get("id")
                        # quiz에서 원본 문제 찾기
                        orig_q = next((q for q in quiz_questions if str(q.get("id")) == str(q_id)), {})
                        question_text = orig_q.get("question", item.get("question", "문제 없음"))
                        
                        st.markdown(f"""
                        **Q{q_id}.** {question_text[:80]}...
                        - 내 답: {item.get('user_answer', '?')} ❌
                        - 정답: {item.get('correct_answer', '?')} ✅
                        """)
                
                if st.button("🚀 반복 학습 시작!", type="primary", use_container_width=True):
                    # 취약점 분석 추가
                    analyzed_wrong = []
                    for item in wrong_items:
                        q_id = str(item.get("id"))
                        orig_q = next((q for q in quiz_questions if str(q.get("id")) == q_id), {})
                        analyzed = WeaknessAnalyzer.analyze_wrong_answer(
                            orig_q, 
                            item.get("user_answer", ""),
                            item.get("correct_answer", "")
                        )
                        analyzed["why_correct_ko"] = item.get("why_correct_ko", "")
                        analyzed["why_user_wrong_ko"] = item.get("why_user_wrong_ko", "")
                        analyzed_wrong.append(analyzed)
                    
                    RepeatLearningManager.start_repeat_learning(analyzed_wrong, quiz_questions)
                    st.rerun()
            else:
                st.warning("⚠️ 먼저 퀴즈를 풀고 채점을 받아주세요. 틀린 문제가 있어야 반복 학습이 가능합니다.")
        
        # 진행 중인 경우
        else:
            # 진행 상황 표시
            st.markdown("#### 📊 진행 상황")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("완료", f"{progress['completed']}/{progress['total']}")
            with col2:
                st.metric("남은 문제", progress['remaining'])
            with col3:
                st.metric("총 시도", progress['total_retries'])
            
            st.progress(progress['progress_percent'] / 100, text=f"진행률: {progress['progress_percent']}%")
            
            # 완료 체크
            if RepeatLearningManager.is_complete():
                st.balloons()
                st.success("🎉 모든 문제를 정복했습니다! 훌륭해요!")
                
                # 완료 통계
                state = st.session_state.get(RepeatLearningManager.SESSION_KEY, {})
                completed = state.get("completed", [])
                
                st.markdown("#### 📊 반복 학습 결과")
                for item in completed:
                    retries = item.get("retries_needed", 1)
                    emoji = "🌟" if retries == 1 else "✅" if retries <= 3 else "💪"
                    st.markdown(f"{emoji} Q{item.get('id')}: {retries}번 만에 성공")
                
                col_restart, col_end = st.columns(2)
                with col_restart:
                    if st.button("🔄 처음부터 다시", key="repeat_restart"):
                        RepeatLearningManager.reset()
                        st.rerun()
                with col_end:
                    if st.button("🏠 반복 학습 종료", key="repeat_end"):
                        RepeatLearningManager.reset()
                        st.rerun()
            else:
                # 현재 문제 풀기
                current_q = RepeatLearningManager.get_next_question()
                
                if current_q:
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
                    st.markdown(f"**{current_q.get('question', '문제 로딩 중...')}**")
                    
                    # 선택지
                    choices = current_q.get("choices", {})
                    
                    with st.form(f"repeat_answer_form_{q_id}"):
                        answer = st.radio(
                            "답을 선택하세요",
                            options=["A", "B", "C", "D"],
                            format_func=lambda x: f"{x}. {choices.get(x, '')}",
                            horizontal=True,
                            index=None,
                            key=f"repeat_answer_{q_id}"
                        )
                        
                        col_submit, col_similar = st.columns([2, 1])
                        with col_submit:
                            submitted = st.form_submit_button("✅ 제출", type="primary", use_container_width=True)
                        with col_similar:
                            gen_similar = st.form_submit_button("🔄 유사 문제", use_container_width=True)
                    
                    # 답안 제출 처리
                    if submitted:
                        if not answer:
                            st.error("답을 선택해주세요!")
                        else:
                            is_correct, result = RepeatLearningManager.check_answer(answer)
                            
                            if is_correct:
                                st.success(f"🎉 정답입니다! ({result['retry_count']}번 만에 성공)")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ 오답입니다. 정답: {result['correct_answer']}")
                                
                                # 해설 표시
                                why_correct = current_q.get("why_correct_ko", "")
                                if why_correct:
                                    st.info(f"💡 **해설:** {why_correct}")
                                
                                evidence = current_q.get("evidence_quote", "")
                                if evidence:
                                    st.markdown(f"📄 **근거:** _{evidence}_")
                    
                    # 유사 문제 생성
                    if gen_similar:
                        with st.spinner("유사 문제 생성 중..."):
                            similar = generate_similar_question(current_q, model=gen_model)
                            if similar:
                                RepeatLearningManager.replace_with_similar(similar)
                                st.success("✅ 유사 문제가 생성되었습니다!")
                                st.rerun()
                    
                    # 반복 학습 종료 버튼
                    st.divider()
                    if st.button("🛑 반복 학습 중단", key="repeat_stop"):
                        RepeatLearningManager.reset()
                        st.rerun()
    
    # ==========================================
    # 서브탭 3: TTS 섀도잉
    # ==========================================
    with subtab3:
        st.subheader("🗣️ 섀도잉 연습")
        
        # 현재 코칭 결과 확인
        any_coach = (
            st.session_state.get("audio_coach") or 
            st.session_state.get("youtube_coach") or 
            st.session_state.get("text_coach")
        )
        
        if any_coach:
            render_shadowing_section(any_coach)
        else:
            st.info("💡 먼저 퀴즈를 풀고 채점을 받으면 섀도잉 연습 문장이 생성됩니다.")
            
            # 직접 입력 옵션
            st.divider()
            st.markdown("#### ✍️ 직접 입력하여 연습")
            
            custom_text = st.text_area(
                "인도네시아어 문장 입력",
                placeholder="Selamat pagi! Apa kabar?",
                height=100
            )
            
            if custom_text:
                speed = st.selectbox(
                    "재생 속도",
                    options=list(TTS_SPEED_OPTIONS.keys()),
                    format_func=lambda x: TTS_SPEED_OPTIONS[x]["label"],
                    index=2,
                    key="custom_tts_speed"
                )
                render_tts_player(custom_text, "", speed, "custom")
    
    # ==========================================
    # 서브탭 4: 현재 세션 퀴즈 (기존 기능)
    # ==========================================
    with subtab4:
        st.subheader("📝 현재 세션 퀴즈 풀이")
        
        # 교육적 가치 분석 결과 표시
        educational_analysis = st.session_state.get("educational_analysis")
        
        if educational_analysis:
            st.subheader("📋 교육적 가치 분석")
            
            with st.expander("📖 분석 결과 보기", expanded=True):
                st.markdown(f"**주제:** {educational_analysis.get('main_topic', 'N/A')}")
                st.markdown(f"**교육 수준:** {educational_analysis.get('educational_level', 'N/A')}")
                st.markdown(f"**교육적 관련성:** {educational_analysis.get('relevance_score', 'N/A')}/10")
                
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
            st.session_state.get("audio_transcript") or
            st.session_state.get("youtube_transcript") or
            st.session_state.get("extracted_text")
        )
        
        if not available_transcript:
            st.info("📌 먼저 '오디오 학습', 'YouTube 학습', 또는 '텍스트 학습' 탭에서 학습 자료를 준비해주세요.")
        else:
            current_source = st.session_state.get("current_source", "Unknown")
            st.caption(f"**출처:** {current_source}")
            
            # 퀴즈 생성 버튼
            if st.button("🎯 퀴즈 5문항 생성", type="primary", key="btn_generate_quiz"):
                try:
                    # 텍스트가 너무 길면 잘라서 사용
                    quiz_text = available_transcript[:4000] if len(available_transcript) > 4000 else available_transcript
                    prompt = QUIZ_PROMPT.format(
                        transcript=quiz_text,
                        num_questions=num_questions,
                        level=level
                    )
                    
                    if debug:
                        with st.expander("🔍 DEBUG: QUIZ_PROMPT (일부)"):
                            st.code(prompt[:1200])
                    
                    with st.spinner("퀴즈를 생성 중..."):
                        quiz = llm_json(prompt, model=gen_model)
                    
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
                            format_func=lambda k, choices=choices: f"{k}. {choices.get(k, '')}",
                            key=f"q_{qid}",
                            index=None,
                            horizontal=True,
                        )
                        user_answers[str(qid)] = pick if pick else ""
                        
                        st.divider()
                    
                    submitted = st.form_submit_button("✅ 채점하고 학습 플랜 받기", type="primary")
                
                # 채점 및 코칭
                if submitted:
                    # 모든 답안이 선택되었는지 확인
                    empty_answers = [qid for qid, ans in user_answers.items() if not ans]
                    if empty_answers:
                        st.error(f"⚠️ 모든 문제에 답을 선택해주세요! (미선택 문제: {', '.join(['Q' + qid for qid in empty_answers])})")
                    else:
                        try:
                            condition_simple = condition.split()[0] if condition else "B"
                            
                            # user_answers를 session_state에 저장 (payload에서 사용하기 위함)
                            st.session_state["tab4_user_answers"] = user_answers
                            
                            prompt = COACH_PROMPT.format(
                                transcript=available_transcript[:4000],
                                quiz_json=json.dumps(quiz, ensure_ascii=False),
                                user_answers=json.dumps(user_answers, ensure_ascii=False),
                                condition=condition_simple,
                            )
                            
                            if debug:
                                with st.expander("🔍 DEBUG: COACH_PROMPT (일부)"):
                                    st.code(prompt[:1200])
                            
                            with st.spinner("코칭 결과를 생성 중... (Structured Outputs 사용)"):
                                # Structured Outputs 사용
                                coach = llm_structured(prompt, CoachResponse, model=gen_model)
                                coach = sanitize_coach_structured(coach, quiz, user_answers)
                            
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
            col3.metric("컨디션", condition.split()[0] if condition else "미설정")
            
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
    # 서브탭 5: SRS 간격 반복 복습
    # ==========================================
    with subtab5:
        st.subheader("📅 간격 반복 복습 (Spaced Repetition)")
        st.info("💡 틀린 문제가 자동으로 SRS에 등록되어, 최적의 시간에 복습할 수 있습니다.")
        
        # SRS 통계
        srs_stats = SpacedRepetitionSystem.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 항목", srs_stats["total_items"])
        with col2:
            st.metric("오늘 복습", srs_stats["due_today"], delta="예정" if srs_stats["due_today"] > 0 else None)
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
                cat_info = CEFR_CATEGORIES.get(category, {"icon": "📌", "name": category})
                
                with st.expander(f"{cat_info['icon']} {question}...", expanded=(i == 0)):
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
                            SpacedRepetitionSystem.record_review(item['id'], is_correct=True, quality=4)
                            st.success("✅ 다음 복습은 나중에!")
                            st.rerun()
                    with col_bad:
                        if st.button("❌ 헷갈려요", key=f"srs_bad_{item['id']}"):
                            SpacedRepetitionSystem.record_review(item['id'], is_correct=False, quality=2)
                            st.warning("🔄 내일 다시 복습!")
                            st.rerun()
        else:
            st.success("🎉 오늘 복습할 항목이 없습니다! 훌륭해요!")
            
            if srs_stats["total_items"] == 0:
                st.info("💡 퀴즈에서 틀린 문제가 자동으로 SRS에 추가됩니다. 퀴즈를 풀어보세요!")
        
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

# =====================================================
# TAB 5: 설정
# =====================================================

with tab5:
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
    col4.metric("총 용량", f"{total_size / 1024 / 1024:.1f} MB", help="전체 로그 폴더 용량")
    
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
                    st.caption(f"📄 {os.path.basename(file)} ({file_size:.1f} KB) - {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    st.caption(f"📄 {os.path.basename(file)}")
            
            if len(upload_wav) > display_count:
                st.caption(f"... 외 {len(upload_wav) - display_count}개")
            
            st.divider()
            
            # 삭제 버튼
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🗑️ 모든 임시 오디오 삭제", type="secondary", use_container_width=True):
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
                        st.success(f"✅ {deleted_count}개 파일 삭제 완료 ({deleted_size / 1024 / 1024:.2f} MB 절약)")
                        st.rerun()
                    else:
                        st.warning("삭제된 파일이 없습니다.")
            
            with col_btn2:
                # 오래된 파일만 삭제 (7일 이전)
                old_wav = [f for f in upload_wav 
                           if datetime.fromtimestamp(os.path.getmtime(f)) < datetime.now() - timedelta(days=7)]
                
                if st.button(f"🗑️ 7일 이전 파일 삭제 ({len(old_wav)}개)", 
                             type="secondary", 
                             use_container_width=True,
                             disabled=len(old_wav)==0):
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
                        st.success(f"✅ {deleted_count}개 파일 삭제 완료 ({deleted_size / 1024 / 1024:.2f} MB 절약)")
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
                    date_key = mtime.strftime('%Y-%m-%d')
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
                old_logs_30 = [f for f in log_json 
                               if datetime.fromtimestamp(os.path.getmtime(f)) < datetime.now() - timedelta(days=30)]
                
                if st.button(f"🗑️ 30일 이전 로그 삭제 ({len(old_logs_30)}개)", 
                             type="secondary", 
                             use_container_width=True,
                             disabled=len(old_logs_30)==0):
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
                old_logs_7 = [f for f in log_json 
                              if datetime.fromtimestamp(os.path.getmtime(f)) < datetime.now() - timedelta(days=7)]
                
                if st.button(f"🗑️ 7일 이전 로그 삭제 ({len(old_logs_7)}개)", 
                             type="secondary", 
                             use_container_width=True,
                             disabled=len(old_logs_7)==0):
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
                        st.caption(f"📄 {os.path.basename(file)} ({file_size:.1f} KB) - {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col_download:
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            st.download_button(
                                label="💾",
                                data=file_content,
                                file_name=os.path.basename(file),
                                mime="application/json",
                                key=f"download_result_{i}",
                                use_container_width=True
                            )
                        except:
                            pass
                except:
                    st.caption(f"📄 {os.path.basename(file)}")
            
            if len(result_json) > display_count:
                st.caption(f"... 외 {len(result_json) - display_count}개")
            
            st.divider()
            
            # 백업 및 삭제
            st.warning("⚠️ **주의**: 결과 파일을 삭제하면 복구할 수 없습니다. 백업 후 삭제하세요.")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                # 모든 결과 파일 백업 (ZIP)
                if st.button("📦 모든 결과 백업 (ZIP)", type="primary", use_container_width=True):
                    import zipfile
                    
                    backup_name = f"backup_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    backup_path = os.path.join(LOG_DIR, backup_name)
                    
                    try:
                        with zipfile.ZipFile(backup_path, 'w') as zipf:
                            for file in result_json:
                                zipf.write(file, os.path.basename(file))
                        
                        st.success(f"✅ 백업 완료: {backup_name}")
                        
                        # 다운로드 버튼 제공
                        with open(backup_path, 'rb') as f:
                            st.download_button(
                                label="📥 백업 파일 다운로드",
                                data=f,
                                file_name=backup_name,
                                mime="application/zip",
                                key="download_backup",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"백업 실패: {e}")
            
            with col_btn2:
                # 30일 이전 결과 삭제
                old_results = [f for f in result_json 
                               if datetime.fromtimestamp(os.path.getmtime(f)) < datetime.now() - timedelta(days=30)]
                
                if st.button(f"🗑️ 30일 이전 결과 삭제 ({len(old_results)}개)", 
                             type="secondary", 
                             use_container_width=True,
                             disabled=len(old_results)==0):
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
        st.markdown("""
        **버전 정보**
        - 앱 버전: v1.0.0
        - Python: """ + f"{os.sys.version.split()[0]}" + """
        - Streamlit: """ + f"{st.__version__}" + """
        """)
    
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
            help="7일 이상 된 임시 오디오 파일을 자동으로 삭제합니다"
        )
        
        if auto_clean_enabled:
            st.info("💡 다음 앱 실행 시 자동 정리가 활성화됩니다. (현재 세션에서는 설정만 저장됩니다)")
        
        st.divider()
        
        st.markdown("**캐시 초기화**")
        
        if st.button("🗑️ Streamlit 캐시 초기화", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ 캐시가 초기화되었습니다. ASR 모델이 다시 로드됩니다.")
            st.rerun()

# =====================================================
# 푸터
# =====================================================

st.divider()
st.caption("""
**🔒 개인정보 보호 및 저작권 준수**
- YouTube 영상은 임베드 형태로만 제공되며, 자동 다운로드하지 않습니다.
- 웹 크롤링은 공개된 교육 자료에 한해 제공되며, 저작권을 준수합니다.
- 생성된 퀴즈 및 코칭 내용은 원본 텍스트를 1:1 복사하지 않고 재작성됩니다.
""")