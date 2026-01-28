# api/quiz_profile.py
from __future__ import annotations
from typing import Any, Dict

from core.level import get_level_profile


def build_quiz_profile(level: str, num_questions: int) -> Dict[str, Any]:
    """
    app.py에서 만들던 profile dict를 FastAPI용으로 재구성.
    core.quiz.generate_quiz(profile=...)에 그대로 넣어줍니다.
    """
    base_profile = get_level_profile(level)

    profile: Dict[str, Any] = {
        "level": base_profile.level,                 # 원래 코드 유지
        "max_questions": num_questions,
        "target_vocab_band": base_profile.target_vocab_band,
        "num_questions": num_questions,
        "cefr_level": level,
        # app.py에서는 difficulty_profile_prompt(level, num_questions) 넣었는데
        # 우선 최소 동작을 위해 문자열을 간단히 구성 (템플릿이 level_profile을 쓰는 경우 대비)
        "level_profile": f"CEFR={level}, num_questions={num_questions}, target_vocab_band={base_profile.target_vocab_band}",
    }

    # app.py에서 마지막에 덮어쓰던 동작 유지
    profile["level"] = level
    return profile

