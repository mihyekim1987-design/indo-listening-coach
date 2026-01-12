# prompts.py
# -*- coding: utf-8 -*-
"""
프롬프트 템플릿 모음
- EDUCATIONAL_ANALYSIS_PROMPT: 교육적 가치 분석
- QUIZ_PROMPT: 퀴즈 생성
- COACH_PROMPT: 채점 및 코칭
"""

# =====================================================
# 1. 교육적 가치 분석 프롬프트
# =====================================================

EDUCATIONAL_ANALYSIS_PROMPT = """역할: 당신은 교육 콘텐츠 분석 전문가입니다.

입력: [TRANSCRIPT]는 비디오 또는 텍스트 자료입니다.

목표: 제공된 TRANSCRIPT를 분석하여 교육적 가치를 평가하고, 주요 학습 포인트를 식별하세요.

# Step by Step Instructions
1. TRANSCRIPT를 주의 깊게 읽으세요.
2. 주요 주제 또는 테마를 식별하세요 (반복되는 키워드, 구문, 개념 파악).
3. TRANSCRIPT를 교육 콘텐츠의 다양한 측면을 나타내는 세그먼트로 나누세요.
4. 각 세그먼트에서 가르치거나 논의되는 핵심 정보, 개념, 기술을 요약하세요.
5. 주요 학습 포인트를 식별하고 나열하세요 (사실, 원리, 이론, 기술 등).
6. 전체 교육적 관련성을 평가하세요 (명확성, 깊이, 학습 가능성 고려).
7. 교육 콘텐츠와 주요 학습 포인트를 요약한 보고서를 생성하세요.
8. 보고서를 검토하여 TRANSCRIPT의 교육적 가치를 정확히 반영하는지 확인하세요.

필수 규칙:
- 출력은 반드시 JSON만 반환 (추가 설명 금지)
- main_topic: 주요 주제 (한국어)
- educational_level: 교육 수준 (A1, A2, B1, B2 등)
- relevance_score: 교육적 관련성 점수 (1-10)
- key_learning_points: 주요 학습 포인트 목록 (한국어, 3-5개)
- summary: 콘텐츠 요약 (한국어, 2-3문장)

출력 JSON 스키마:
{{
  "main_topic": "string",
  "educational_level": "string",
  "relevance_score": 0,
  "key_learning_points": ["string", "string", "string"],
  "summary": "string",
  "segments": [
    {{
      "heading": "string",
      "key_concepts": ["string", "string"]
    }}
  ]
}}

[TRANSCRIPT]
{video_transcript}

IMPORTANT NOTE: Start directly with the JSON output. Do not output any delimiters or explanations.
"""

# =====================================================
# 2. 퀴즈 생성 프롬프트
# =====================================================

QUIZ_PROMPT = """역할: 당신은 인도네시아어 평가 출제자입니다.

입력: 
- [TRANSCRIPT]는 학습용 오디오 또는 텍스트를 받아쓴 내용입니다.
- [NUM_QUESTIONS]는 생성할 문제 수입니다.
- [LEVEL]는 학습 수준입니다 (초급: A1~A2, 중급: B1~B2).

목표: TRANSCRIPT를 바탕으로 {level} 수준의 4지선다 문제 {num_questions}개를 만드세요.

필수 규칙:
- 출력은 반드시 JSON만 반환(추가 설명 금지)
- 각 문제는 TRANSCRIPT에서 정답 근거를 찾을 수 있어야 함
- 각 문제마다 evidence_quote는 TRANSCRIPT에서 그대로 인용(10~20단어)
- 유형 비율 (문제 수에 따라 조정):
  * 3문제: FACT 1개, DETAIL 1개, GIST 1개
  * 5문제: FACT 2개, DETAIL 2개, GIST 1개
  * 10문제: FACT 4개, DETAIL 4개, GIST 2개
- 오답은 그럴듯하되 TRANSCRIPT와 명확히 불일치해야 함
- explanation_id는 인도네시아어로 1~2문장
- 초급(A1~A2): 쉬운 단어 사용, 짧은 문장
- 중급(B1~B2): 다양한 어휘, 복잡한 문장 구조 가능
- questions는 반드시 {num_questions}개, id는 1~{num_questions}

출력 JSON 스키마(예시, 그대로 구조를 따르되 내용만 바꿀 것):
{{
  "mode": "BIPA_LISTENING_BEGINNER",
  "level": "A1",
  "questions": [
    {{
      "id": 1,
      "type": "FACT",
      "question": "string",
      "choices": {{"A":"string","B":"string","C":"string","D":"string"}},
      "answer": "A",
      "evidence_quote": "string",
      "explanation_id": "string"
    }}
  ],
  "key_vocab": [
    {{"word":"string","meaning_ko":"string","example_id":"string","example_ko":"string"}}
  ]
}}

[TRANSCRIPT]
{transcript}

IMPORTANT NOTE: Start directly with the JSON output. Do not output any delimiters or explanations.
"""

# =====================================================
# 3. 채점 및 코칭 프롬프트
# =====================================================

COACH_PROMPT = """역할: 당신은 인도네시아어 초급(A1~A2) 학습 코치입니다.

입력:
- [QUIZ_JSON] 문제/정답/해설 JSON (answer 필드가 정답)
- [USER_ANSWERS] 사용자의 답안 (예: {{"1": "A", "2": "B", ...}})
- [TRANSCRIPT] 원문 텍스트
- [CONDITION] 사용자의 컨디션: A=여유/ B=보통/ C=힘듦

목표:
1) 각 문항에 대해 해설 제공 (is_correct는 QUIZ_JSON의 answer와 USER_ANSWERS 비교로 판단)
2) 취약 포인트 3개 (한국어)
3) 내일 10분 학습 플랜 (컨디션 반영)
4) shadowing 문장 3개

중요:
- 각 문항의 is_correct 판단: QUIZ_JSON의 answer와 USER_ANSWERS를 비교 (대소문자 무시)
- 예: answer가 "B"이고 user가 "B"면 is_correct: true, "A"면 is_correct: false

출력은 JSON만 반환(추가 설명 금지)

출력 JSON 스키마:
{{
  "items": [
    {{
      "id": 1,
      "is_correct": true,
      "correct_explain_ko": "왜 정답이 맞는지 설명",
      "wrong_reason_ko": "오답인 경우 왜 틀렸는지 설명 (정답이면 빈 문자열)",
      "choice_notes_ko": {{"A": "A 설명", "B": "B 설명", "C": "C 설명", "D": "D 설명"}},
      "evidence_quote": "TRANSCRIPT에서 근거 인용"
    }}
  ],
  "weak_points_ko": ["취약점1", "취약점2", "취약점3"],
  "tomorrow_plan_10min_ko": [
    {{"minute": "0-2", "task": "과제"}},
    {{"minute": "2-6", "task": "과제"}},
    {{"minute": "6-9", "task": "과제"}},
    {{"minute": "9-10", "task": "과제"}}
  ],
  "shadowing_sentences": [
    {{"id": "Indonesian sentence", "ko": "한국어 번역"}}
  ]
}}

[TRANSCRIPT]
{transcript}

[QUIZ_JSON]
{quiz_json}

[USER_ANSWERS]
{user_answers}

[CONDITION]
{condition}

IMPORTANT NOTE: Start directly with the JSON output. Do not output any delimiters or explanations.
"""

# =====================================================
# 4. CEFR 취약점 카테고리 정의
# =====================================================

CEFR_WEAKNESS_CATEGORIES = {
    "vocabulary": {
        "name_ko": "어휘",
        "name_id": "Kosakata",
        "description": "단어 의미, 동의어, 반의어, 문맥상 어휘 선택",
        "cefr_descriptors": ["A1: 기본 어휘", "A2: 일상 어휘", "B1: 확장 어휘", "B2: 추상적 어휘"]
    },
    "grammar_tense": {
        "name_ko": "시제",
        "name_id": "Tense/Kala",
        "description": "sudah, sedang, akan, belum 등 시제 표현",
        "cefr_descriptors": ["A1: 현재/과거 기본", "A2: 미래/완료", "B1: 복합 시제"]
    },
    "grammar_affix": {
        "name_ko": "접사",
        "name_id": "Imbuhan",
        "description": "me-, ber-, di-, -kan, -i, -an 등 접사 활용",
        "cefr_descriptors": ["A1: me-/ber- 기본", "A2: di-/-kan 수동", "B1: 복합 접사"]
    },
    "numbers": {
        "name_ko": "숫자/수량",
        "name_id": "Angka/Bilangan",
        "description": "숫자, 날짜, 시간, 수량 표현",
        "cefr_descriptors": ["A1: 기본 숫자", "A2: 날짜/시간", "B1: 복잡한 수량"]
    },
    "honorifics": {
        "name_ko": "경어/존칭",
        "name_id": "Bahasa Hormat",
        "description": "Bapak, Ibu, Anda, kamu 등 존칭 및 격식체",
        "cefr_descriptors": ["A1: 기본 호칭", "A2: 상황별 격식", "B1: 미묘한 뉘앙스"]
    },
    "comprehension": {
        "name_ko": "독해/청해",
        "name_id": "Pemahaman",
        "description": "텍스트 이해, 핵심 정보 파악, 추론",
        "cefr_descriptors": ["A1: 단순 사실", "A2: 세부 정보", "B1: 추론/요지"]
    },
    "sentence_structure": {
        "name_ko": "문장 구조",
        "name_id": "Struktur Kalimat",
        "description": "어순, 접속사, 관계절 등",
        "cefr_descriptors": ["A1: 단문", "A2: 복문 기초", "B1: 복잡한 문장"]
    },
    "context": {
        "name_ko": "문맥 이해",
        "name_id": "Konteks",
        "description": "상황에 맞는 표현 선택, 문맥상 의미 파악",
        "cefr_descriptors": ["A2: 기본 문맥", "B1: 함축적 의미", "B2: 미묘한 차이"]
    }
}

# =====================================================
# 5. 오답 복습 퀴즈 생성 프롬프트
# =====================================================

REMEDIAL_QUIZ_PROMPT = """역할: 당신은 인도네시아어 교육 전문가입니다.

학습자가 틀린 문제를 분석하여, 같은 취약점을 보완할 유사 문제를 생성합니다.

입력:
- [WRONG_QUESTIONS]: 학습자가 틀린 문제 목록 (원문, 정답, 오답, 취약 카테고리 포함)
- [TRANSCRIPT]: 원본 텍스트 (참고용)
- [LEVEL]: 학습자 수준

CEFR 취약점 카테고리:
- vocabulary: 어휘 (단어 의미, 동의어, 반의어)
- grammar_tense: 시제 (sudah, sedang, akan, belum)
- grammar_affix: 접사 (me-, ber-, di-, -kan, -i)
- numbers: 숫자/수량 (숫자, 날짜, 시간)
- honorifics: 경어/존칭 (Bapak, Ibu, Anda)
- comprehension: 독해/청해 (핵심 정보, 추론)
- sentence_structure: 문장 구조 (어순, 접속사)
- context: 문맥 이해 (상황별 표현)

목표:
1) 각 틀린 문제의 취약점 카테고리를 분석
2) 해당 카테고리에 맞는 유사 문제 생성 (같은 유형이지만 다른 내용)
3) 난이도는 동일하거나 살짝 쉽게 출제

출력 JSON 스키마:
{{
  "weakness_analysis": [
    {{
      "original_question_id": 1,
      "category": "vocabulary",
      "category_ko": "어휘",
      "specific_weakness": "동사 의미 혼동"
    }}
  ],
  "remedial_questions": [
    {{
      "id": 1,
      "related_to_original_id": 1,
      "category": "vocabulary",
      "question": "다음 중 'membaca'의 의미로 올바른 것은?",
      "choices": {{"A": "쓰다", "B": "읽다", "C": "듣다", "D": "말하다"}},
      "answer": "B",
      "explanation_ko": "'membaca'는 '읽다'라는 뜻입니다.",
      "hint_ko": "me- + baca(읽다) = membaca"
    }}
  ]
}}

[WRONG_QUESTIONS]
{wrong_questions}

[TRANSCRIPT]
{transcript}

[LEVEL]
{level}

IMPORTANT NOTE: Start directly with the JSON output. Do not output any delimiters or explanations.
"""

