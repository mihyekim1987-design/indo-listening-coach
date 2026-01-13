# prompts.py
# -*- coding: utf-8 -*-
"""
프롬프트 템플릿 모음 (format-safe 버전)
- EDUCATIONAL_ANALYSIS_PROMPT
- QUIZ_PROMPT
- COACH_PROMPT
- REMEDIAL_QUIZ_PROMPT
- SIMILAR_QUESTION_PROMPT
- AI_LEARNING_COACH_PROMPT
- CEFR_WEAKNESS_CATEGORIES
주의:
- 이 파일의 프롬프트들은 app.py에서 .format(...)을 사용하므로
  프롬프트 내부의 JSON 예시/스키마 중괄호는 반드시 {{ }} 로 이스케이프 처리되어야 함.
"""

# =====================================================
# 1. 교육적 가치 분석 프롬프트
# =====================================================

EDUCATIONAL_ANALYSIS_PROMPT = """역할: 당신은 교육 콘텐츠 분석 전문가입니다.

입력: [TRANSCRIPT]는 비디오 또는 텍스트 자료입니다.

목표: 제공된 TRANSCRIPT를 분석하여 교육적 가치를 평가하고, 주요 학습 포인트를 식별하세요.

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
{transcript}

IMPORTANT NOTE: Start directly with the JSON output. Do not output any delimiters or explanations.
"""


# =====================================================
# 2. 퀴즈 생성 프롬프트
# - ✅ 문제/선택지/해설을 "인도네시아어"로 생성하도록 변경
# - evidence_quote는 TRANSCRIPT에서 그대로 인용(인니어 원문)
# =====================================================

QUIZ_PROMPT = """역할: 당신은 인도네시아어(Indonesian) 평가 출제자입니다.

입력:
- [TRANSCRIPT]는 학습용 오디오 또는 텍스트를 받아쓴 인도네시아어 원문입니다.
- [NUM_QUESTIONS]는 생성할 문제 수입니다.
- [LEVEL]는 학습 수준입니다 (A1~A2, B1~B2).

목표:
TRANSCRIPT를 바탕으로 {level} 수준의 4지선다 문제 {num_questions}개를 만드세요.

필수 규칙:
- 출력은 반드시 JSON만 반환(추가 설명 금지)
- question, choices, explanation은 모두 "인도네시아어"로 작성
- evidence_quote는 TRANSCRIPT에서 그대로 인용 (10~20 단어 권장, 인도네시아어 원문)
- 각 문제는 TRANSCRIPT에서 정답 근거를 찾을 수 있어야 함
- 오답은 그럴듯하되 TRANSCRIPT와 명확히 불일치해야 함
- questions는 반드시 {num_questions}개, id는 1~{num_questions}
- 초급(A1~A2): 쉬운 단어, 짧은 문장
- 중급(B1~B2): 다양한 어휘, 복잡한 문장 구조 가능

출력 JSON 스키마:
{{
  "mode": "BIPA_LISTENING",
  "level": "A1",
  "questions": [
    {{
      "id": 1,
      "type": "FACT|DETAIL|GIST",
      "question": "string (Indonesian)",
      "choices": {{"A":"string","B":"string","C":"string","D":"string"}},
      "answer": "A|B|C|D",
      "evidence_quote": "string (Indonesian quote from transcript)",
      "explanation": "string (Indonesian)"
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
# - format-safe(중괄호 이스케이프 완료)
# - app.py가 why_correct_ko 등을 사용하므로 필드 유지
# =====================================================

COACH_PROMPT = """역할: 당신은 인도네시아어 초급(A1~A2) 학습 코치입니다.

입력:
- [QUIZ_JSON] 문제/정답/해설 JSON (answer 필드가 정답)
- [USER_ANSWERS] 사용자의 답안 JSON (예: {{"1":"A","2":"B"}})
- [TRANSCRIPT] 원문 텍스트(인도네시아어)
- [CONDITION] 사용자의 컨디션: A=여유 / B=보통 / C=힘듦

목표:
1) 각 문항에 대해 채점 (is_correct는 QUIZ_JSON의 answer와 USER_ANSWERS 비교로 판단, 대소문자 무시)
2) 각 문항 해설 제공 (한국어)
3) 취약 포인트 3개 (한국어)
4) 내일 10분 학습 플랜 (컨디션 반영, 한국어)
5) shadowing 문장 3개 (인도네시아어 원문 + 한국어 번역)

중요:
- is_correct 판단은 비교 결과로만 결정
- evidence_quote는 TRANSCRIPT에서 인용 (인도네시아어 원문 유지)

출력은 JSON만 반환(추가 설명 금지)

출력 JSON 스키마:
{{
  "items": [
    {{
      "id": 1,
      "is_correct": true,
      "correct_explain_ko": "왜 정답이 맞는지 설명",
      "wrong_reason_ko": "오답인 경우 왜 틀렸는지 설명 (정답이면 빈 문자열)",
      "choice_notes_ko": {{"A":"A 설명","B":"B 설명","C":"C 설명","D":"D 설명"}},
      "evidence_quote": "TRANSCRIPT에서 근거 인용(인도네시아어)"
    }}
  ],
  "wrong_items": [
    {{
      "id": 1,
      "question": "문제 텍스트",
      "user_answer": "A",
      "correct_answer": "B",
      "why_correct_ko": "정답 해설(한국어)",
      "why_user_wrong_ko": "오답 해설(한국어)",
      "choices_explanation": {{"A":"A 해설","B":"B 해설","C":"C 해설","D":"D 해설"}},
      "evidence_quote": "TRANSCRIPT 근거 문장(인도네시아어)"
    }}
  ],
  "score": {{"correct": 0, "total": 5, "percent": 0}},
  "weak_points_ko": ["취약점1", "취약점2", "취약점3"],
  "tomorrow_plan_10min_ko": [
    {{"minute":"0-2","task":"과제"}},
    {{"minute":"2-6","task":"과제"}},
    {{"minute":"6-9","task":"과제"}},
    {{"minute":"9-10","task":"과제"}}
  ],
  "shadowing_sentences": [
    {{"id":"Indonesian sentence", "ko":"한국어 번역"}}
  ]
}}

[TRANSCRIPT]
{transcript}

[QUIZ_JSON]
{quiz_json}

[USER_ANSWERS]
{user_answers_json}

[CONDITION]
{condition}

IMPORTANT NOTE: Start directly with the JSON output. Do not output any delimiters or explanations.
"""


# =====================================================
# 4. CEFR 취약점 카테고리 정의
# (app.py가 이 이름으로 import 할 가능성이 높아 그대로 유지)
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
# - format-safe
# =====================================================

REMEDIAL_QUIZ_PROMPT = """역할: 당신은 인도네시아어 교육 전문가입니다.

학습자가 틀린 문제를 분석하여, 같은 취약점을 보완할 유사 문제를 생성합니다.

입력:
- [WRONG_QUESTIONS]: 학습자가 틀린 문제 목록 (원문, 정답, 오답, 취약 카테고리 포함)
- [TRANSCRIPT]: 원본 텍스트 (참고용)
- [LEVEL]: 학습자 수준

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
      "choices": {{"A":"쓰다","B":"읽다","C":"듣다","D":"말하다"}},
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


# =====================================================
# 6. 유사 문제 생성 (단일 문제) 프롬프트
# - ✅ “문제/선택지/해설을 인도네시아어”로 바꾸고 싶다는 요구 반영
# - format-safe
# =====================================================

SIMILAR_QUESTION_PROMPT = """You are an Indonesian language education expert.

Create ONE similar but different question based on the original question below.

Original Question:
- Question: {question}
- Category: {category}
- Correct Answer: {correct_answer}
- Evidence Quote: {evidence_quote}

CRITICAL REQUIREMENTS:
1. Keep the same category and difficulty level
2. Test the same grammar/vocabulary concept but use different sentences/situations
3. Question and choices MUST be in Indonesian
4. evidence_quote MUST be Indonesian ONLY (no Korean characters)
5. Create a completely new Indonesian sentence for evidence_quote that tests the same concept

RESPOND ONLY in this JSON format (no other text):
{{
  "id": 99,
  "question": "Pertanyaan baru (Indonesian only)",
  "category": "{category}",
  "choices": {{"A":"...","B":"...","C":"...","D":"..."}},
  "answer": "A|B|C|D",
  "evidence_quote": "Kalimat bahasa Indonesia (Indonesian only)",
  "explanation": "Penjelasan singkat dalam bahasa Indonesia"
}}
"""


# =====================================================
# 7. AI 학습 코치 프롬프트
# - app.py import 체크에서 요구하므로 반드시 존재해야 함
# - format-safe
# =====================================================

AI_LEARNING_COACH_PROMPT = """당신은 인도네시아어 학습 전문 AI 코치입니다.

학습자의 퀴즈 결과를 분석하고 맞춤형 학습 조언을 제공하세요.

학습자 정보:
- 정답률: {score_percent}% ({correct}/{total}문제)
- 컨디션: {condition}
- 틀린 문제 수: {wrong_count}개
- 주요 취약 카테고리: {weak_categories}

틀린 문제 상세:
{wrong_details}

요구사항:
1. 학습자의 현재 수준을 파악
2. 실행 가능한 학습 전략 제시
3. 긍정적이고 격려하는 톤 유지
4. 단기(오늘~내일)와 중기(1주일) 학습 계획 제시
5. 취약한 부분 중심으로 제안하되 강점도 언급

출력 형식 (JSON):
{{
  "overall_assessment": "전반 평가(한국어 2-3문장)",
  "strengths": ["강점 1", "강점 2"],
  "weaknesses": ["약점 1", "약점 2", "약점 3"],
  "immediate_actions": [
    {{
      "action": "구체적인 행동",
      "reason": "이유",
      "time_needed": "소요 시간"
    }}
  ],
  "weekly_plan": [
    {{
      "day": "Day 1-2",
      "focus": "집중 영역",
      "activities": ["활동 1", "활동 2"]
    }},
    {{
      "day": "Day 3-4",
      "focus": "집중 영역",
      "activities": ["활동 1", "활동 2"]
    }},
    {{
      "day": "Day 5-7",
      "focus": "집중 영역",
      "activities": ["활동 1", "활동 2"]
    }}
  ],
  "motivational_message": "격려 메시지(한국어 2-3문장)",
  "recommended_resources": [
    {{
      "type": "리소스 유형",
      "name": "리소스 이름",
      "description": "설명"
    }}
  ]
}}

IMPORTANT NOTE: Start directly with the JSON output. Do not output any delimiters or explanations.
"""
