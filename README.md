# 🇮🇩 Belajar Bahasa Indonesia

AI 기반 인도네시아어 학습 앱 - Streamlit으로 제작

## ✨ 주요 기능

### 1. 📊 CEFR 기반 난이도 선택
- A1 (입문): 기본 인사, 자기소개
- A2 (초급): 일상 대화, 기본 문법
- B1 (중급): 의견 표현, 복잡한 문법
- B2 (중상급): 추상적 주제, 고급 문법

### 2. 💪 컨디션별 문제 수 조절
- **A (여유)**: 10문항 - 충분한 시간으로 깊이 있는 학습
- **B (보통)**: 5문항 - 적당한 분량의 균형 잡힌 학습
- **C (힘듦)**: 3문항 - 핵심만 빠르게 학습

### 3. 📚 다양한 학습 콘텐츠
- **텍스트**: 위키문헌, VOA 뉴스 기사
- **유튜브**: VOA Indonesia 다큐멘터리, 뉴스 (자막 추출)
- **오디오**: Whisper ASR로 음성 인식 학습
- **직접 입력**: 원하는 URL 직접 입력 가능

### 4. 🎯 AI 퀴즈 생성
- GPT-4o가 콘텐츠 기반 퀴즈 자동 생성
- 카테고리별 분류 (어휘/문법/독해/청해)
- **근거 원문 인용(evidence_quote)** 포함
- 상세 해설 제공

### 5. 📄 임베드 미리보기
- 학습 전 원본 콘텐츠 임베드로 확인
- 텍스트 자동 추출
- 유튜브 영상 임베드 + 자막 추출

### 6. 🔄 틀린 문제 반복 학습
- 틀린 문제와 유사한 새로운 문제 생성
- 정답할 때까지 반복 연습

### 7. 📋 내일을 위한 10분 학습 플랜
- AI 코치의 개인화된 학습 계획
- 취약 영역 분석 및 개선 방법
- 복습할 단어 및 문법 팁 제공

### 8. 📊 학습 결과 기록
- 모든 학습 기록 저장
- 통계 및 진행 상황 확인
- JSON 파일 다운로드 가능

---

## 🚀 시작하기

### 1. 필요 조건
- Python 3.8 이상
- OpenAI API 키 (GPT-4o 접근 권한)

### 2. 설치

```bash
# 레포지토리 클론 또는 파일 다운로드
cd indonesian_app

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 3. API 키 설정

`.env.example` 파일을 `.env`로 복사하고 API 키를 입력하세요:

```bash
cp .env.example .env
```

`.env` 파일 편집:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
HF_TOKEN=your-huggingface-token-here  # 선택사항
```

### 4. 오디오 파일 준비 (선택)

`indo-listening-coach` 폴더에 오디오 샘플 파일을 넣어주세요:
- `sample_A.wav`
- `sample_B.wav`

### 5. 실행

```bash
streamlit run app.py
```

---

## 📁 프로젝트 구조

```
indonesian_app/
├── app.py                    # 메인 Streamlit 앱
├── requirements.txt          # 의존성 패키지
├── .env.example              # 환경변수 예시
├── .env                      # 실제 API 키 (생성 필요)
├── .gitignore               # Git 제외 파일
├── README.md                # 프로젝트 설명
├── indo-listening-coach/    # 오디오 샘플 폴더
│   ├── sample_A.wav
│   └── sample_B.wav
├── components/              # UI 컴포넌트 (확장용)
├── utils/                   # 유틸리티 함수 (확장용)
└── data/                    # 학습 데이터 (확장용)
```

---

## 🎨 디자인

- 모던하고 깔끔한 교육 앱 스타일
- 파란색/보라색 그라데이션 테마
- 반응형 카드 레이아웃
- 직관적인 네비게이션

---

## 🔧 기술 스택

- **Frontend**: Streamlit
- **AI**: OpenAI GPT-4o
- **ASR**: Hugging Face Transformers (Sparkplugx1904/whisper-base-id)
- **웹 스크래핑**: BeautifulSoup, requests
- **유튜브**: youtube-transcript-api
- **스타일링**: Custom CSS
- **데이터**: JSON (로컬 저장)

---

## 📝 학습 흐름

```
홈 → 난이도 선택 → 콘텐츠 선택 → 미리보기(임베드) 
                                    ↓
     학습 결과 ← 학습 플랜 ← 채점 결과 ← 퀴즈 풀기
        ↓
   틀린 문제 복습 (반복)
```

---

## ⚠️ 주의사항

1. **API 키 보안**: `.env` 파일은 절대 GitHub에 업로드하지 마세요!
2. **오디오 파일**: Whisper 모델은 처음 실행 시 다운로드됩니다 (시간 소요)
3. **유튜브 자막**: 자막이 없는 영상은 텍스트 추출이 불가능합니다
4. **웹 스크래핑**: 일부 웹사이트는 텍스트 추출이 제한될 수 있습니다

---

## 🌟 향후 개선 예정

- [ ] TTS 섀도잉 연습
- [ ] Spaced Repetition 알고리즘
- [ ] 사용자 인증 및 프로필
- [ ] 주간/월간 학습 리포트
---

## 📄 라이선스

학습 콘텐츠 출처:
- 위키문헌: 퍼블릭 도메인
- VOA Indonesia: CC BY / 퍼블릭 도메인

---

Made with ❤️ for Indonesian language learners