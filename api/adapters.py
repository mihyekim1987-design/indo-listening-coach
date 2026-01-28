# api/adapters.py
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from core.asr import transcribe_audio
from core.quiz import generate_quiz
from core.grading import grade_quiz


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def run_asr(
    asr_pipe: Any,
    audio_bytes: bytes,
    filename: str,
    target_sr: int = 16000,
    language: str = "indonesian",
) -> str:
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(audio_bytes)
        f.flush()
        return transcribe_audio(asr_pipe, f.name, target_sr=target_sr, language=language)

def _norm_choice(s: str) -> str:
    # 공백/대소문자 차이로 인한 중복을 잡기 위한 정규화
    return " ".join(str(s).strip().lower().split())


def _validate_quiz_payload_no_dup_choices(payload: dict) -> list[str]:
    """
    quiz payload에서 '선택지 중복'을 검출.
    - choices 값이 4개 모두 동일/부분중복이면 issues에 기록
    - answer 키가 choices에 없으면 issues에 기록
    """
    issues: list[str] = []
    questions = payload.get("questions", [])
    if not isinstance(questions, list) or not questions:
        return ["no_questions"]

    for q in questions:
        qid = q.get("id", "?")
        choices = q.get("choices", {})
        if not isinstance(choices, dict) or not choices:
            issues.append(f"qid={qid}:missing_choices")
            continue

        # 값 중복 체크
        vals = [_norm_choice(v) for v in choices.values() if v is not None]
        if len(vals) >= 2 and len(set(vals)) != len(vals):
            issues.append(f"qid={qid}:duplicate_choice_values")

        # 정답 키가 choices에 존재하는지 체크
        ans = str(q.get("answer", "")).strip()
        if ans and ans not in choices:
            issues.append(f"qid={qid}:answer_not_in_choices({ans})")

    return issues


def run_quiz(
    transcript: str,
    level: Optional[str] = None,
    extra_constraint: Optional[str] = None,
) -> Dict[str, Any]:

    import os
    from openai import OpenAI

    from api.config import QUIZ_MODEL, QUIZ_PROMPT, DEFAULT_LEVEL, DEFAULT_NUM_QUESTIONS
    from api.quiz_profile import build_quiz_profile

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. (export OPENAI_API_KEY=... 필요)")

    client = OpenAI(api_key=api_key)

    lvl = level or DEFAULT_LEVEL
    profile = build_quiz_profile(lvl, DEFAULT_NUM_QUESTIONS)

    EXTRA_CONSTRAINT = """
[CONSTRAINT - MUST FOLLOW]
- For every question, all 4 choices must be DIFFERENT strings.
- Do NOT repeat the same choice text across A/B/C/D.
- Answer must be one of the choice keys (A/B/C/D).
"""

    MAX_TRIES = 3
    last_issues: list[str] | None = None
    out: Dict[str, Any] | None = None

    for attempt in range(1, MAX_TRIES + 1):
        template_used = QUIZ_PROMPT + "\n" + EXTRA_CONSTRAINT
        if extra_constraint and str(extra_constraint).strip():
            template_used += "\n" + str(extra_constraint).strip()

        quiz_result = generate_quiz(
            client=client,
            model=QUIZ_MODEL,
            template=template_used,
            transcript=transcript,
            profile=profile,
        )

        payload = getattr(quiz_result, "payload", None) or (
            quiz_result.get("payload") if isinstance(quiz_result, dict) else None
        )

        if not isinstance(payload, dict):
            last_issues = [f"attempt={attempt}:payload_not_dict"]
            continue

        issues = _validate_quiz_payload_no_dup_choices(payload)
        if not issues:
            out = _to_dict(quiz_result)
            break

        last_issues = [f"attempt={attempt}:{x}" for x in issues]

    if out is None:
        raise RuntimeError(f"Quiz validation failed after {MAX_TRIES} tries: {last_issues}")

    return out


def run_grading(quiz: Dict[str, Any], user_answers: Dict[str, Any]) -> Dict[str, Any]:
    import inspect

    # ✅ grader가 보통 "qid"를 문자열로 다루므로 키를 문자열로 고정
    if isinstance(user_answers, dict):
        user_answers = {str(k): v for k, v in user_answers.items()}

    sig = inspect.signature(grade_quiz)
    param_names = list(sig.parameters.keys())

    # ✅ 첫 인자가 answers 쪽이면 순서를 바꿔 호출
    answers_first = param_names and param_names[0] in {
        "user_answers", "answers", "submitted_answers", "responses"
    }

    try:
        if answers_first:
            out = grade_quiz(user_answers, quiz)
        else:
            out = grade_quiz(quiz, user_answers)
    except TypeError:
        # 최후의 안전망: positional 그대로
        out = grade_quiz(quiz, user_answers)

    out = _to_dict(out)
    return out if isinstance(out, dict) else {"result": out}


