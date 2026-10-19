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
        logger.debug("Could not fetch document list", exc_info=True)
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
                for skipped in result.get("skipped", []):
                    st.warning(f"Skipped `{skipped['filename']}`: {skipped['reason']}")
                phi_counts = result.get("phi_entities_pseudonymized") or {}
                if phi_counts:
                    summary = ", ".join(f"{n} {kind.lower()}" for kind, n in sorted(phi_counts.items()))
                    st.info(f"De-identified before indexing: {summary}.")
                # Invalidate collection cache so doc count refreshes
                st.session_state.pop("collections", None)
            except api_client.ApiError as exc:
                st.error(f"Upload failed: {exc.detail}")
            except Exception:
                logger.error("Upload failed", exc_info=True)
                st.error("Upload failed. Check logs for details.")
