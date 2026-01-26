# core/quiz.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import string
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
            {
                "role": "system",
                "content": "You must output ONLY valid JSON. No extra text, no markdown, no code blocks.",
            },
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


def build_quiz_prompt(template: str, transcript: str, profile: Dict[str, Any]) -> str:
    ctx = dict(profile or {})
    ctx["transcript"] = transcript
    ctx.setdefault("num_questions", 5)

    fields = [
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    ]
    required = set(fields)

    if "level" in required:
        ctx.setdefault("level", ctx.get("cefr_level", "A2"))
    if "cefr_level" in required:
        ctx.setdefault("cefr_level", ctx.get("level", "A2"))

    missing = [field for field in required if field not in ctx]
    if missing:
        missing_sorted = sorted(missing)
        raise ValueError(
            "Missing prompt fields: "
            f"{missing_sorted}. Add these keys to the profile dict passed to "
            "build_quiz_prompt."
        )

    return template.format(**ctx)


def _normalize_profile(profile: Any) -> LevelProfile:
    if isinstance(profile, LevelProfile):
        return profile
    if isinstance(profile, dict):
        level = profile.get("level") or profile.get("cefr_level") or "A2"
        max_questions = (
            profile.get("num_questions") or profile.get("max_questions") or 5
        )
        try:
            max_questions = int(max_questions)
        except (TypeError, ValueError):
            max_questions = 5
        return LevelProfile(
            level=level,
            max_questions=max_questions,
            target_vocab_band=profile.get("target_vocab_band"),
        )
    return LevelProfile(level="A2", max_questions=5)


def generate_quiz(
    client: Any, model: str, template: str, transcript: str, profile: Dict[str, Any]
) -> QuizResult:
    prompt = build_quiz_prompt(template, transcript, profile)
    result = call_openai_json(client, model, prompt)
    validate_quiz_payload(result.payload, _normalize_profile(profile))
    return result
