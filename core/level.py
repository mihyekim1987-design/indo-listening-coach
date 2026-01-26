# core/level.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional

CEFRLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]

@dataclass(frozen=True)
class LevelProfile:
    level: CEFRLevel
    max_questions: int = 5
    target_vocab_band: Optional[str] = None

def get_level_profile(level: CEFRLevel) -> LevelProfile:
    if level in ("A1", "A2"):
        return LevelProfile(level=level, max_questions=5, target_vocab_band="basic")
    if level in ("B1", "B2"):
        return LevelProfile(level=level, max_questions=5, target_vocab_band="intermediate")
    return LevelProfile(level=level, max_questions=5, target_vocab_band="advanced")

def validate_quiz_payload(payload: Dict[str, Any], profile: LevelProfile) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Quiz payload must be a JSON object (dict).")
    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Quiz payload must include non-empty 'questions' list.")
    if len(questions) > profile.max_questions:
        raise ValueError(f"Too many questions: {len(questions)} > {profile.max_questions}")
