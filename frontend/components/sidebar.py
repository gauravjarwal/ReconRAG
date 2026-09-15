from __future__ import annotations

import logging

import streamlit as st

import api_client

logger = logging.getLogger(__name__)


def render_sidebar() -> str | None:
    """
    Render the collection management sidebar.
    Returns the currently selected collection name, or None if none exist.
    """
    st.sidebar.title("ReconRAG")
    st.sidebar.markdown("---")

    # Refresh button
    if st.sidebar.button("Refresh Collections"):
        st.session_state.pop("collections", None)

    # Load collections
    if "collections" not in st.session_state:
        try:
            st.session_state.collections = api_client.list_collections()
        except Exception:
            logger.error("Could not load collections", exc_info=True)
            st.sidebar.error("Could not load collections. Check backend connectivity.")
            st.session_state.collections = []

    collections: list[dict] = st.session_state.collections
    collection_names = [c["name"] for c in collections]

    # Create new collection
    st.sidebar.subheader("Create Collection")
    new_name = st.sidebar.text_input("Collection name", placeholder="my-documents")
    if st.sidebar.button("Create"):
        if not new_name.strip():
            st.sidebar.warning("Please enter a collection name.")
        else:
            try:
                api_client.create_collection(new_name.strip())
                st.sidebar.success(f"Created '{new_name}'")
                st.session_state.pop("collections", None)
                st.rerun()
            except Exception:
                logger.error("Failed to create collection '%s'", new_name, exc_info=True)
                st.sidebar.error("Failed to create collection. It may already exist.")

    st.sidebar.markdown("---")

    # Select collection
    st.sidebar.subheader("Active Collection")
    if not collection_names:
        st.sidebar.info("No collections yet. Create one above.")
        return None

    selected = st.sidebar.selectbox(
        "Select collection",
        options=collection_names,
        index=0,
    )

    # Show doc count
    doc_count = next((c["document_count"] for c in collections if c["name"] == selected), 0)
    st.sidebar.caption(f"{doc_count} chunk(s) indexed")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Retrieval Settings")
    st.session_state.reranking_enabled = st.sidebar.toggle(
        "Enable Re-ranking",
        value=st.session_state.get("reranking_enabled", False),
        help="Re-ranking improves accuracy but adds ~3-5s latency. Disable for faster responses.",
    )

    return selected
