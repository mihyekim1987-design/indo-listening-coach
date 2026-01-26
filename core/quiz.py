# core/quiz.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import json
from .level import LevelProfile, validate_quiz_payload

@dataclass
class QuizResult:
    payload: Dict[str, Any]
    raw_text: str

def call_openai_json(client: Any, model: str, prompt: str) -> QuizResult:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You must output ONLY valid JSON. No extra text, no markdown, no code blocks."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError("Empty response content from OpenAI.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}") from e
    return QuizResult(payload=payload, raw_text=raw)

def build_quiz_prompt(template: str, transcript: str, profile: LevelProfile) -> str:
    return template.format(
        transcript=transcript,
        level=profile.level,
        max_questions=profile.max_questions,
        vocab_band=profile.target_vocab_band or "",
    )

def generate_quiz(client: Any, model: str, template: str, transcript: str, profile: LevelProfile) -> QuizResult:
    prompt = build_quiz_prompt(template, transcript, profile)
    result = call_openai_json(client, model, prompt)
    validate_quiz_payload(result.payload, profile)
    return result
