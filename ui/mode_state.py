from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import streamlit as st

MODES = ("audio", "video", "text", "speaking")


def _default_repeat_state() -> Dict:
    return {
        "wrong_queue": [],
        "current_question": None,
        "retry_count": {},
        "completed": [],
        "total_retries": 0,
        "active": False,
        "celebration_shown": False,
    }


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    for sep in (". ", "? ", "! "):
        if sep in cleaned:
            return cleaned.split(sep)[0].strip() + sep.strip()
    return cleaned


def get_mode_state(mode: str) -> Dict:
    if "learn_state" not in st.session_state:
        st.session_state["learn_state"] = {}

    learn_state = st.session_state["learn_state"]
    if mode not in learn_state:
        learn_state[mode] = {
            "shadowing_items": [],
            "wrong_items": [],
            "wrong_question_ids": [],
            "retry_quiz": None,
            "retry_active": False,
            "retry_completed": False,
            "retry_mastered": False,
            "last_result": None,
            "last_score": None,
            "last_attempt": {},
            "review_summary": {},
            "repeat_learning": _default_repeat_state(),
            "updated_at": datetime.now().isoformat(),
        }
    return learn_state[mode]


def reset_mode_ephemeral(mode: str) -> None:
    state = get_mode_state(mode)
    state["shadowing_items"] = []
    state["wrong_items"] = []
    state["wrong_question_ids"] = []
    state["retry_quiz"] = None
    state["retry_active"] = False
    state["retry_completed"] = False
    state["retry_mastered"] = False
    state["repeat_learning"] = _default_repeat_state()
    state["updated_at"] = datetime.now().isoformat()


def set_shadowing_from_wrong(mode: str, wrong_items: List[Dict]) -> None:
    state = get_mode_state(mode)
    shadowing_items = []
    seen = set()

    for item in wrong_items:
        question_id = str(item.get("id") or item.get("question_id") or "").strip()
        evidence = (item.get("evidence_quote") or "").strip()
        snippet = (
            item.get("transcript_snippet")
            or item.get("source_text")
            or item.get("transcript")
            or item.get("question")
            or ""
        )
        sentence = evidence or _first_sentence(snippet)
        if not sentence:
            continue

        dedupe_key = question_id or sentence
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        shadowing_items.append(
            {
                "question_id": question_id or None,
                "sentence": sentence,
                "evidence_quote": evidence,
                "source_snippet": snippet,
                "timestamp": item.get("timestamp") or datetime.now().isoformat(),
                "mode": mode,
            }
        )

    state["shadowing_items"] = shadowing_items
    state["updated_at"] = datetime.now().isoformat()


def record_mode_result(mode: str, result: Dict) -> None:
    state = get_mode_state(mode)
    timestamp = result.get("timestamp") or datetime.now().isoformat()
    wrong_items = result.get("wrong_items", [])

    state["last_result"] = result
    state["last_score"] = result.get("score")
    state["last_attempt"] = {
        "timestamp": timestamp,
        "score": result.get("score"),
        "source": result.get("source", mode),
    }
    state["wrong_items"] = wrong_items
    state["wrong_question_ids"] = [
        str(item.get("id") or item.get("question_id"))
        for item in wrong_items
        if item.get("id") or item.get("question_id")
    ]
    state["review_summary"] = {
        "score": result.get("score", {}),
        "wrong_count": len(wrong_items),
        "source": result.get("source", mode),
        "timestamp": timestamp,
        "condition": result.get("condition"),
        "level": result.get("level"),
    }
    set_shadowing_from_wrong(mode, wrong_items)
