"""ReconRAG UI."""

from __future__ import annotations

import api_client
import streamlit as st
from components.chat import render_chat
from components.library import render_library
from components.sidebar import render_sidebar

st.set_page_config(page_title="ReconRAG", layout="wide")

if not api_client.is_ready():
    st.error("The API isn't ready. Check that the API and Postgres are running.")
    st.stop()

collection = render_sidebar()
chat_tab, library_tab = st.tabs(["Ask", "Documents"])
with chat_tab:
    render_chat(collection)
with library_tab:
    render_library(collection)
