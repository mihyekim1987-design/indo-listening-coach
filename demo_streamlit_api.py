import os
import time
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Listening Coach (YouTube Demo)", layout="centered")
st.title("Listening Coach Demo (YouTube → FastAPI)")
st.caption(f"API_BASE_URL = {API_BASE_URL}")

# -------------------------
# API helpers
# -------------------------
def fetch_yt_transcript(url: str, lang: str = "id") -> dict:
    payload = {"url": url, "lang": lang}
    r = requests.post(f"{API_BASE_URL}/yt/transcript", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()  # {"video_id": "...", "transcript": "..."}

def repeat_similar(transcript: str, level: str, prev_quiz_payload: dict):
    payload = {"transcript": transcript, "level": level, "prev_quiz": prev_quiz_payload}
    r = requests.post(f"{API_BASE_URL}/repeat/similar", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["quiz"]["payload"]

def create_quiz(transcript: str, level: str):
    payload = {"transcript": transcript, "level": level}
    r = requests.post(f"{API_BASE_URL}/quiz", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["quiz"]["payload"]

def grade_attempt(quiz_payload, user_answers: dict):
    payload = {"quiz": quiz_payload, "user_answers": user_answers}
    r = requests.post(f"{API_BASE_URL}/attempts", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def render_explanations(qp: dict, user_answers: dict, result: dict):
    questions = qp.get("questions", [])
    total = len(questions)

    raw = result.get("raw") or {}
    score = raw.get("score", result.get("score", 0))

    st.subheader("결과 해설")
    st.success(f"점수: {score}/{total}")

    wrong_items = result.get("wrong_items") or raw.get("wrong_items") or []
    wrong_map = {str(w.get("qid")): w for w in wrong_items}

    for q in questions:
        qid = str(q.get("id"))
        q_text = q.get("question", "")
        choices = q.get("choices", {}) or {}
        correct_key = str(q.get("answer"))

        user_key = user_answers.get(qid)
        is_correct = (user_key == correct_key)

        title = f"Q{qid}. {'✅ 정답' if is_correct else '❌ 오답'}"
        with st.expander(title, expanded=not is_correct):
            st.markdown(f"**문제**: {q_text}")

            def label(k):
                if not k:
                    return "(미응답)"
                return f"{k}. {choices.get(k,'')}"

            st.markdown(f"- **내 답**: {label(user_key)}")
            st.markdown(f"- **정답**: {label(correct_key)}")

            evidence = q.get("evidence_quote") or ""
            if evidence:
                st.caption(f"근거: {evidence}")

            expl = None
            if not is_correct:
                expl = (wrong_map.get(qid) or {}).get("explanation")
            if not expl:
                expl = q.get("explanation") or ""

            if expl:
                st.markdown(f"**해설**: {expl}")

    with st.expander("DEBUG: raw response", expanded=False):
        st.json(result)

# -------------------------
# UX helpers
# -------------------------
def reset_quiz_state(keep_transcript: bool = True):
    # 퀴즈/답안 UI 초기화
    if not keep_transcript:
        st.session_state.pop("transcript", None)
        st.session_state.pop("video_id", None)
        st.session_state.pop("yt_url", None)

    st.session_state.pop("quiz_payload", None)
    st.session_state.pop("grade_result", None)

    st.session_state.pop("review_only", None)
    st.session_state.pop("wrong_qids", None)

    # 라디오 선택값 초기화를 위해 nonce 증가
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1

def validate_youtube_url(u: str) -> bool:
    u = (u or "").strip()
    return u.startswith("http://") or u.startswith("https://")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("설정")
    level = st.selectbox("레벨", ["A1", "A2", "B1", "B2"], index=1)
    lang = st.selectbox("자막 우선 언어", ["id", "en"], index=0)
    st.divider()
    if st.button("전체 초기화"):
        reset_quiz_state(keep_transcript=False)
        st.success("초기화 완료")

# -------------------------
# Main: Step 1) YouTube link → transcript
# -------------------------
st.subheader("1) YouTube 링크로 자막 가져오기")

yt_url = st.text_input(
    "YouTube URL",
    value=st.session_state.get("yt_url", ""),
    placeholder="예: https://www.youtube.com/watch?v=Bu74xgFtyyY",
)

# ✅ 입력값이 있으면 우선, 없으면 세션값
embed_url = yt_url.strip() if yt_url else (st.session_state.get("yt_url") or "")
if embed_url and validate_youtube_url(embed_url):
    st.video(embed_url)
    st.caption(f"자막 우선 언어: {lang} | 레벨: {level}")


col1, col2 = st.columns([1, 1])

with col1:
    if st.button("자막 가져오기", type="primary"):
        if not validate_youtube_url(yt_url):
            st.error("올바른 YouTube URL을 입력해 주세요. (http/https로 시작)")
        else:
            try:
                with st.spinner("YouTube 자막을 가져오는 중..."):
                    data = fetch_yt_transcript(yt_url, lang=lang)
                    st.session_state["yt_url"] = yt_url
                    st.session_state["video_id"] = data.get("video_id")
                    st.session_state["transcript"] = data.get("transcript", "").strip()
                    reset_quiz_state(keep_transcript=True)
                st.success("자막 추출 완료")
            except requests.HTTPError as e:
                # FastAPI에서 400 detail 내려온 경우 포함
                st.error(f"자막 추출 실패: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"자막 추출 실패: {str(e)}")

with col2:
    if st.button("자막/퀴즈 초기화"):
        reset_quiz_state(keep_transcript=False)
        st.info("초기화했습니다. 새 링크를 입력해 주세요.")

if "video_id" in st.session_state and st.session_state.get("video_id"):
    st.caption(f"video_id = {st.session_state['video_id']}")

if "transcript" in st.session_state and st.session_state["transcript"]:
    st.text_area("Transcript", st.session_state["transcript"], height=220)

# -------------------------
# Step 2) transcript → quiz
# -------------------------
st.subheader("2) 퀴즈 생성")

if "transcript" not in st.session_state or not st.session_state.get("transcript"):
    st.info("먼저 1)에서 YouTube 링크로 자막을 가져와 주세요.")
else:
    if st.button("퀴즈 생성"):
        try:
            with st.spinner("퀴즈 생성 중..."):
                quiz_payload = create_quiz(st.session_state["transcript"], level)
                st.session_state["quiz_payload"] = quiz_payload
                st.session_state.pop("grade_result", None)
                st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
            st.success("퀴즈 생성 완료")
        except requests.HTTPError as e:
            st.error(f"퀴즈 생성 실패: {e.response.text if e.response is not None else str(e)}")
        except Exception as e:
            st.error(f"퀴즈 생성 실패: {str(e)}")

# -------------------------
# Step 3) answer + grade (no pre-checked answers)
# -------------------------
if "quiz_payload" in st.session_state:
    qp = st.session_state["quiz_payload"]
    st.subheader("3) 문제 풀기")

    # nonce로 라디오 key를 바꾸면 기본선택/이전선택이 남는 문제를 방지
    nonce = st.session_state.get("nonce", 0)

    # ✅ 오답만 복습 모드 (기본 False)
    review_only = st.session_state.get("review_only", False)  # True/False
    wrong_qids_set = set(map(str, st.session_state.get("wrong_qids", []) or []))  # {"1","3",...}

    # ✅ 표시할 문항 리스트 확정
    questions_all = qp.get("questions", []) or []
    if review_only and wrong_qids_set:
        questions = [q for q in questions_all if str(q.get("id")) in wrong_qids_set]
    else:
        questions = questions_all

    # 복습 모드인데 오답이 없으면 안내
    if review_only and not questions:
        st.info("복습할 오답이 없습니다. 🎯 (전체 문제로 다시 풀어보려면 복습 모드를 해제하세요.)")
    else:
        with st.form(key=f"quiz_form_{nonce}"):
            answers: dict[str, str] = {}
            unanswered: list[str] = []

            for q in questions:
                qid = str(q.get("id"))
                st.markdown(f"**Q{qid}. {q.get('question','')}**")

                choice_items = list((q.get("choices") or {}).items())
                labels = [f"{k}. {v}" for k, v in choice_items]
                keys = [k for k, _ in choice_items]

                # ✅ 기본 선택 제거: index=None
                sel_label = st.radio(
                    "선택",
                    labels,
                    index=None,
                    key=f"qsel_{nonce}_{qid}",
                    label_visibility="collapsed",
                )

                if sel_label is None:
                    unanswered.append(qid)
                else:
                    answers[qid] = keys[labels.index(sel_label)]

                evidence = q.get("evidence_quote") or ""
                if evidence:
                    st.caption(f"evidence: {evidence}")
                st.divider()

            submitted = st.form_submit_button("제출/채점")

        # --- 채점 처리 ---
        if submitted:
            if unanswered:
                st.warning(f"아직 선택하지 않은 문항이 있습니다: {', '.join(unanswered)}")
            else:
                try:
                    with st.spinner("채점 중..."):
                        result = grade_attempt(qp, answers)

                    # ✅ 결과/해설 렌더링을 위해 저장
                    st.session_state["grade_result"] = result
                    st.session_state["grade_answers"] = answers
                    st.session_state["grade_qp"] = qp

                    # ✅ 오답 문항 id 저장 (복습 모드용)
                    raw = result.get("raw") or {}
                    wrong_items = result.get("wrong_items") or raw.get("wrong_items") or []
                    wrong_qids = [
                        str(w.get("qid"))
                        for w in wrong_items
                        if w.get("qid") is not None
                    ]
                    st.session_state["wrong_qids"] = wrong_qids

                    # 점수 표시
                    score = result.get("score")
                    st.success(f"점수: {score}")

                except requests.HTTPError as e:
                    st.error(f"채점 실패: {e.response.text if e.response is not None else str(e)}")
                except Exception as e:
                    st.error(f"채점 실패: {str(e)}")


# -------------------------
# Result display (해설 렌더링)
# -------------------------
if "grade_result" in st.session_state:
    qp_for_result = st.session_state.get("grade_qp") or st.session_state.get("quiz_payload")
    answers_for_result = st.session_state.get("grade_answers", {})

    # render_explanations()가 파일에 이미 정의되어 있다는 전제
    render_explanations(qp_for_result, answers_for_result, st.session_state["grade_result"])

    # ✅ CTA: 오답이 있으면 복습 퀴즈 모드로 전환
    wrong_qids = st.session_state.get("wrong_qids", []) or []
    if wrong_qids:
        st.divider()
        st.subheader("복습")
        st.caption(f"오답 문항: {', '.join(map(str, wrong_qids))}")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("오답만 다시 풀기", type="primary"):
                st.session_state["review_only"] = True
                st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
                st.info("복습 모드 ON: 오답 문항만 다시 풉니다.")
                st.rerun()

        with col_b:
            if st.button("전체 다시 풀기"):
                st.session_state["review_only"] = False
                st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
                st.info("복습 모드 OFF: 전체 문항을 다시 풉니다.")
                st.rerun()

