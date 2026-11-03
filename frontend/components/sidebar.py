from __future__ import annotations

import api_client
import streamlit as st


def render_sidebar() -> dict | None:
    """Returns the selected collection, or None if there are none yet."""
    st.sidebar.title("ReconRAG")
    collections = api_client.list_collections()
    names = [c["name"] for c in collections]
    selected = st.sidebar.selectbox("Collection", names) if names else None

    with st.sidebar.expander("New collection", expanded=not names):
        with st.form("new_collection", clear_on_submit=True):
            name = st.text_input("Name", placeholder="policies-2026")
            phi = st.checkbox("Contains patient data (de-identify before indexing)")
            if st.form_submit_button("Create") and name:
                try:
                    api_client.create_collection(name, phi)
                    st.rerun()
                except api_client.ApiError as exc:
                    st.error(exc.message)
    return next((c for c in collections if c["name"] == selected), None)
