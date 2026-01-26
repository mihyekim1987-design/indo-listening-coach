# core/grading.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class WrongItem:
    qid: str
    question: str
    user_answer: str
    correct_answer: str
    explanation: Optional[str] = None


@dataclass
class GradingResult:
    score: int
    total: int
    wrong_items: List[WrongItem]


def grade_quiz(
    quiz_payload: Dict[str, Any], user_answers: Dict[str, str]
) -> GradingResult:
    questions = quiz_payload.get("questions", [])
    total = len(questions)
    score = 0
    wrong: List[WrongItem] = []

    for q in questions:
        qid = str(q.get("id", ""))  # UI에서 쓰는 id 기준
        q_text = str(q.get("question", ""))
        correct = str(q.get("answer", ""))
        explanation = q.get("explanation")

        ua = (user_answers.get(qid, "") or "").strip()

        if ua and ua == correct:
            score += 1
        else:
            wrong.append(
                WrongItem(
                    qid=qid,
                    question=q_text,
                    user_answer=ua,
                    correct_answer=correct,
                    explanation=explanation,
                )
            )

    return GradingResult(score=score, total=total, wrong_items=wrong)
