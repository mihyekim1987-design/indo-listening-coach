# core/grading.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class WrongItem:
    q_idx: int
    question: str
    user_answer: str
    correct_answer: str
    explanation: Optional[str] = None

@dataclass
class GradingResult:
    score: int
    total: int
    wrong_items: List[WrongItem]

def grade_quiz(quiz_payload: Dict[str, Any], user_answers: Dict[str, str]) -> GradingResult:
    questions = quiz_payload.get("questions", [])
    total = len(questions)
    score = 0
    wrong: List[WrongItem] = []

    for i, q in enumerate(questions):
        q_text = str(q.get("question", ""))
        correct = str(q.get("answer", ""))
        explanation = q.get("explanation")
        ua = (user_answers.get(str(i), "") or "").strip()

        if ua and ua == correct:
            score += 1
        else:
            wrong.append(WrongItem(i, q_text, ua, correct, explanation))

    return GradingResult(score=score, total=total, wrong_items=wrong)
