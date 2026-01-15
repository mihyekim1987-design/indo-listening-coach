from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Tuple

import streamlit as st


def inject_global_css(css_path: Path) -> None:
    if not css_path.exists():
        return
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_header(
    current_page: str,
    navigate_to_page: Callable[[str], None],
    navigate_to_home: Callable[[], None],
    logo_path: Path,
    nav_items: Iterable[Tuple[str, str]],
    cta_label: str,
) -> None:
    nav_items = list(nav_items)
    active_label = ""
    for page, label in nav_items:
        if page == current_page:
            active_label = label
            break

    if active_label:
        st.markdown(
            f"""
<style>
/* Active header link */
#app-header div[data-testid="stButton"] > button[aria-label="{active_label}"] {{
  color: var(--primary-coral-red) !important;
  box-shadow: inset 0 -2px 0 var(--primary-coral-red) !important;
}}
</style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div id="header-nav-anchor"></div><div id="app-header">', unsafe_allow_html=True)
    logo_col, nav_col, cta_col = st.columns([2, 7, 2])

    with logo_col:
        if logo_path.exists():
            st.image(str(logo_path), width=140)
        else:
            st.markdown('<div class="logo-fallback">Bisa</div>', unsafe_allow_html=True)

    with nav_col:
        nav_cols = st.columns(len(nav_items))
        for (page, label), col in zip(nav_items, nav_cols):
            with col:
                if st.button(label, key=f"nav_{page}", type="secondary", use_container_width=True):
                    navigate_to_page(page)

    with cta_col:
        if st.button(cta_label, key="cta_make_it_bisa", type="primary"):
            navigate_to_home()
    st.markdown("</div>", unsafe_allow_html=True)
