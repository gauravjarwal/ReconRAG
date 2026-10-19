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
    new_phi = st.sidebar.checkbox(
        "Contains PHI (de-identify before indexing)",
        value=True,
        help="Names, dates, MRNs and other identifiers are replaced with pseudonym tokens before "
        "anything is embedded or sent to an LLM. Uncheck only for non-patient data.",
    )
    if st.sidebar.button("Create"):
        if not new_name.strip():
            st.sidebar.warning("Please enter a collection name.")
        else:
            try:
                api_client.create_collection(new_name.strip(), phi_protection=new_phi)
                st.sidebar.success(f"Created '{new_name}'")
                st.session_state.pop("collections", None)
                st.rerun()
            except api_client.ApiError as exc:
                st.sidebar.error(f"Failed to create collection: {exc.detail}")
            except Exception:
                logger.error("Failed to create collection", exc_info=True)
                st.sidebar.error("Failed to create collection. Check backend connectivity.")

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

    # Show doc count and PHI status
    info = next((c for c in collections if c["name"] == selected), {})
    phi_note = " · PHI-protected" if info.get("phi_protection") else ""
    st.sidebar.caption(f"{info.get('document_count', 0)} chunk(s) indexed{phi_note}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Retrieval Settings")
    st.session_state.reranking_enabled = st.sidebar.toggle(
        "Enable Re-ranking",
        value=st.session_state.get("reranking_enabled", False),
        help="Re-ranking improves accuracy but adds ~3-5s latency. Disable for faster responses.",
    )
    if info.get("phi_protection"):
        st.session_state.reidentify_enabled = st.sidebar.toggle(
            "Show real identifiers",
            value=st.session_state.get("reidentify_enabled", False),
            help="Replace pseudonym tokens with original values. Requires a PHI-access API key; every use is audited.",
        )
    else:
        st.session_state.reidentify_enabled = False

    return selected
