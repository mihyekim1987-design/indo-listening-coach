# api/config.py
import os
import prompts as P  # app.py와 동일

QUIZ_PROMPT = P.QUIZ_PROMPT

# app.py에 QUIZ_MODEL 상수가 없으니 env 기반으로 운영
QUIZ_MODEL = os.getenv("QUIZ_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

DEFAULT_LEVEL = os.getenv("DEFAULT_LEVEL", "A2")
DEFAULT_NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "5"))

