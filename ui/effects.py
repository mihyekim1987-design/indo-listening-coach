from __future__ import annotations

from html import escape as _escape

import streamlit as st
import streamlit.components.v1 as components


def celebrate_confetti(
    key: str | None = None, message: str = "🎉 Great job!", force: bool = False
) -> None:
    """
    Streamlit components.html()은 iframe 안에서 실행됩니다.
    height=0일 경우 confetti 캔버스가 iframe 내부에 생성되어 "안 보이는" 문제가 자주 발생합니다.

    해결:
    - window.parent.document에 fixed canvas를 주입해 전체 화면 오버레이로 confetti 실행
    - run_id로 HTML을 매번 다르게 만들어 렌더를 강제
    - CDN 로드 실패 등 예외 상황은 toast/success로 폴백
    """
    once_key = f"confetti_once_{key}" if key else None
    if once_key and st.session_state.get(once_key) and not force:
        return

    # rerun 환경에서 같은 컴포넌트로 최적화되어 스킵되는 것을 막기 위한 run_id
    run_id_key = f"confetti_run_{key}" if key else "confetti_run_global"
    st.session_state[run_id_key] = st.session_state.get(run_id_key, 0) + 1
    run_id = st.session_state[run_id_key]

    canvas_id = f"st-confetti-canvas-{key or 'global'}"
    safe_message = _escape(message).replace("\n", " ")

    html_code = f"""
    <div id="st-confetti-root-{run_id}"></div>
    <script>
    (function () {{
      const CANVAS_ID = "{canvas_id}";
      const CDN = "https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js";

      function ensureScript(doc, cb) {{
        try {{
          const win = doc.defaultView;
          if (win && typeof win.confetti === "function") {{
            cb();
            return;
          }}

          const existing = doc.querySelector('script[data-st-confetti="1"]');
          if (existing) {{
            const iv = setInterval(() => {{
              const w = doc.defaultView;
              if (w && typeof w.confetti === "function") {{
                clearInterval(iv);
                cb();
              }}
            }}, 50);
            setTimeout(() => clearInterval(iv), 2500);
            return;
          }}

          const s = doc.createElement("script");
          s.src = CDN;
          s.async = true;
          s.setAttribute("data-st-confetti", "1");
          s.onload = cb;
          doc.head.appendChild(s);
        }} catch (e) {{
          // ignore, fallback happens server-side
        }}
      }}

      function fireIn(doc) {{
        const win = doc.defaultView;
        if (!win || typeof win.confetti !== "function") return;

        let canvas = doc.getElementById(CANVAS_ID);
        if (!canvas) {{
          canvas = doc.createElement("canvas");
          canvas.id = CANVAS_ID;
          canvas.style.position = "fixed";
          canvas.style.inset = "0";
          canvas.style.width = "100%";
          canvas.style.height = "100%";
          canvas.style.pointerEvents = "none";
          canvas.style.zIndex = "999999";
          doc.body.appendChild(canvas);
        }}

        const myConfetti = win.confetti.create(canvas, {{ resize: true, useWorker: true }});
        myConfetti({{
          particleCount: 140,
          startVelocity: 32,
          spread: 360,
          ticks: 180,
          origin: {{ x: 0.5, y: 0.3 }}
        }});

        // cleanup
        setTimeout(() => {{
          try {{ canvas.remove(); }} catch(e) {{}}
        }}, 2200);
      }}

      // 1) Prefer parent document overlay (best UX)
      try {{
        const pdoc = window.parent && window.parent.document ? window.parent.document : null;
        if (pdoc) {{
          ensureScript(pdoc, () => fireIn(pdoc));
          return;
        }}
      }} catch (e) {{}}

      // 2) Fallback to iframe document (may not be visible if iframe is tiny)
      ensureScript(document, () => fireIn(document));
    }})();
    </script>
    """

    try:
        # height=1로 "컴포넌트가 실제로 mount"되도록 보장 (0은 브라우저에서 완전히 안 보일 수 있음)
        # Streamlit 버전에 따라 key 인자가 없을 수 있어 안전하게 처리
        try:
            components.html(
                html_code, height=1, scrolling=False, key=f"st_confetti_{key}_{run_id}"
            )
        except TypeError:
            components.html(html_code, height=1, scrolling=False)

        if once_key:
            st.session_state[once_key] = True

    except Exception:
        # 최후의 폴백
        if hasattr(st, "toast"):
            st.toast(message)
        else:
            st.success(message)


def reset_confetti(key: str) -> None:
    """같은 key로 confetti를 다시 실행하고 싶을 때 호출"""
    st.session_state.pop(f"confetti_once_{key}", None)
    st.session_state.pop(f"confetti_run_{key}", None)
