from __future__ import annotations

import logging

import streamlit as st

import api_client

logger = logging.getLogger(__name__)

ALLOWED_TYPES = ["pdf", "docx", "txt", "md"]


def render_uploader(collection_name: str | None) -> None:
    """Render the document upload section."""
    st.subheader("Upload Documents")

    if collection_name is None:
        st.info("Create or select a collection in the sidebar to upload documents.")
        return

    # Show existing documents in this collection
    try:
        existing_docs = api_client.list_documents(collection_name)
    except Exception:
        logger.debug("Could not fetch document list for '%s'", collection_name, exc_info=True)
        existing_docs = []

    if existing_docs:
        st.markdown(f"**Documents in** `{collection_name}` **({len(existing_docs)}):**")
        for doc_name in existing_docs:
            st.markdown(f"- `{doc_name}`")
        st.markdown("---")

    uploaded_files = st.file_uploader(
        f"Upload files to **{collection_name}**",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, MD. Max 10 MB per file.",
    )

    if uploaded_files and st.button("Upload & Ingest", type="primary"):
        with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
            try:
                files = [
                    (f.name, f.read(), f.type or "application/octet-stream")
                    for f in uploaded_files
                ]
                result = api_client.upload_documents(collection_name, files)
                st.success(
                    f"Ingested {result['files_processed']} file(s), "
                    f"{result['chunks_created']} chunk(s) into **{collection_name}**."
                )
                # Invalidate collection cache so doc count refreshes
                st.session_state.pop("collections", None)
            except Exception:
                logger.error("Upload failed for collection '%s'", collection_name, exc_info=True)
                st.error("Upload failed. Check logs for details.")
