"""Upload, live status for every file, and what's indexed."""

from __future__ import annotations

import api_client
import streamlit as st

_ACTIVE = {"queued", "running"}
_ICONS = {
    "queued": "⏳",
    "running": "⚙️",
    "succeeded": "✅",
    "duplicate": "♻️",
    "quarantined": "⚠️",
    "dead": "❌",
}


def _job_rows(jobs: list[dict]) -> list[dict]:
    return [
        {
            "File": job["filename"],
            "Status": f"{_ICONS.get(job['status'], '')} {job['status']}",
            "Stage": job["stage"] if job["status"] in _ACTIVE else "",
            "Chunks": job["stats"].get("chunks"),
            "Why": job["error_detail"] or job["stats"].get("duplicate_of") or "",
        }
        for job in jobs
    ]


@st.fragment(run_every="2s")
def _live_jobs(collection: str) -> None:
    jobs = api_client.jobs(collection)
    if jobs:
        running = sum(job["status"] in _ACTIVE for job in jobs)
        st.caption(f"{running} in progress" if running else "All files processed")
        st.dataframe(_job_rows(jobs), hide_index=True, use_container_width=True)


def _document_rows(documents: list[dict]) -> list[dict]:
    return [
        {
            "Document": doc["title"] or doc["document_key"],
            "File": doc["filename"],
            "Version": doc["version"],
            "Chunks": doc["chunks"],
            "Pages": doc["page_count"],
            "OCR pages": doc["ocr_pages"],
            "Quality": round(doc["quality"], 2),
            "Flags": ", ".join(doc["quality_flags"]),
        }
        for doc in documents
    ]


def render_library(collection: dict | None) -> None:
    if collection is None:
        st.info("Create a collection in the sidebar first.")
        return
    name = collection["name"]
    uploaded = st.file_uploader(
        "Add files: PDF, Word, Excel, CSV, HTML, email, text, Markdown or images",
        accept_multiple_files=True,
    )
    if uploaded and st.button(f"Upload {len(uploaded)} file(s)"):
        try:
            result = api_client.upload(name, [(f.name, f.getvalue()) for f in uploaded])
        except api_client.ApiError as exc:
            st.error(exc.message)
        else:
            for item in result["rejected"]:
                st.warning(f"{item['filename']}: {item['message']}")

    _live_jobs(name)
    st.subheader("Indexed documents")
    documents = api_client.documents(name)
    if documents:
        st.dataframe(_document_rows(documents), hide_index=True, use_container_width=True)
    else:
        st.caption("Nothing indexed yet.")
