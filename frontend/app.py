from __future__ import annotations

import streamlit as st

import api_client
from components.chat import render_chat
from components.sidebar import render_sidebar
from components.uploader import render_uploader

st.set_page_config(
    page_title="ReconRAG",
    page_icon="",
    layout="wide",
)

# Backend health check
if not api_client.health_check():
    st.error("Cannot reach the backend API. Make sure the backend service is running.")
    st.stop()

# Sidebar: collection management
active_collection = render_sidebar()

# Main area: two tabs
tab_upload, tab_chat = st.tabs(["Upload Documents", "Chat"])

with tab_upload:
    render_uploader(active_collection)

with tab_chat:
    render_chat(active_collection)
