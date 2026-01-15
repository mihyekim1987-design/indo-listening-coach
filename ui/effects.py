from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def celebrate_confetti(key: str | None = None, message: str = "🎉 Great job!") -> None:
    flag_key = f"confetti_once_{key}" if key else None
    if flag_key and st.session_state.get(flag_key):
        return

    try:
        components.html(
            """
            <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
            <script>
                (function() {
                    confetti({
                        particleCount: 120,
                        startVelocity: 30,
                        spread: 360,
                        ticks: 140,
                        origin: { x: 0.5, y: 0.3 }
                    });
                })();
            </script>
            """,
            height=0,
            scrolling=False,
        )
        if flag_key:
            st.session_state[flag_key] = True
    except Exception:
        if hasattr(st, "toast"):
            st.toast(message)
        else:
            st.success(message)
