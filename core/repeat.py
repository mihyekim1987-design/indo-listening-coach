# core/repeat.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal
from .grading import WrongItem

RepeatMode = Literal["wrong_repeat", "similar_repeat"]

@dataclass
class RepeatSeed:
    mode: RepeatMode
    prompt_seed: str

def build_repeat_seeds(wrong_items: List[WrongItem], mode: RepeatMode) -> List[RepeatSeed]:
    seeds: List[RepeatSeed] = []
    for wi in wrong_items:
        if mode == "wrong_repeat":
            seed = f"Repeat the concept: {wi.question}. Explain why '{wi.correct_answer}' is correct."
        else:
            seed = f"Create a similar question to: {wi.question} (same difficulty/grammar focus)."
        seeds.append(RepeatSeed(mode=mode, prompt_seed=seed))
    return seeds
