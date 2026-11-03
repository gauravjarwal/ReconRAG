"""Streaming chat. Citations show the file, section and page, and open the source passage."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import api_client
import streamlit as st

_WARNINGS = {
    "unverified_numbers": "Some numbers aren't in the cited sources: {numbers}.",
    "citations_removed": "A citation to a source that doesn't exist was removed.",
    "no_citations": "This answer cites no sources, so treat it with care.",
    "phi_redacted": "An identifier not found in the sources was redacted.",
    "truncated": "The answer was cut short.",
    "answers_disabled": "No model is configured, so here are the best passages instead.",
}


def _location(source: dict[str, Any]) -> str:
    parts = [source["filename"]]
    if source["section"]:
        parts.append(" › ".join(source["section"]))
    if source["page_start"]:
        last = source["page_end"]
        pages = (
            source["page_start"]
            if last == source["page_start"]
            else f"{source['page_start']}-{last}"
        )
        parts.append(f"p. {pages}")
    return " · ".join(parts)


def _render_sources(
    collection: str, sources: list[dict[str, Any]], cited: list[str], key: str
) -> None:
    for source in sources:
        used = " (not cited)" if cited and source["ref"] not in cited else ""
        with st.expander(f"[{source['ref']}] {_location(source)}{used}"):
            if source["injection_risk"]:
                st.warning("This passage looks like a prompt injection attempt.")
            st.markdown(source["text"])
            if source["also_in"]:
                st.caption("Same text also in: " + ", ".join(source["also_in"]))
            if st.button("Show surrounding text", key=f"{key}-{source['ref']}"):
                detail = api_client.chunk(collection, source["chunk_id"])
                for text in detail["before"]:
                    st.caption(text)
                st.markdown(f"**{detail['text']}**")
                for text in detail["after"]:
                    st.caption(text)


def _render_notes(final: dict[str, Any]) -> None:
    for warning in final.get("warnings", []):
        numbers = ", ".join(final.get("unverified_numbers", []))
        st.warning(_WARNINGS.get(warning, warning).format(numbers=numbers))
    for item in final.get("degraded", []):
        st.caption(f"Degraded: {item}")
    model = final.get("model")
    total = final.get("stages_ms", {}).get("total")
    if model:
        cost = model["cost_usd"]
        price = f"${cost:.5f}" if cost is not None else "cost unknown"
        st.caption(
            f"{model['model']} · first token {model['ttft_ms']:.0f} ms · "
            f"total {total:.0f} ms · {price}"
        )


def _render_turn(collection: str, turn: dict[str, Any], key: str) -> None:
    st.chat_message("user").markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"] or "")
        _render_notes(turn["final"])
        _render_sources(collection, turn["sources"], turn["final"].get("cited", []), key)


def _ask(collection: str, question: str, reidentify: bool) -> dict[str, Any]:
    turn: dict[str, Any] = {"question": question, "sources": [], "final": {}, "answer": ""}
    st.chat_message("user").markdown(question)

    def tokens() -> Iterator[str]:
        for event, data in api_client.ask_stream(collection, question, reidentify):
            if event == "sources":
                turn["sources"] = data["sources"]
            elif event == "token":
                yield data["text"]
            elif event == "done":
                turn["final"] = data
            elif event == "error":
                raise api_client.ApiError(503, data["message"])

    with st.chat_message("assistant"):
        streamed = st.write_stream(tokens())
    turn["answer"] = turn["final"].get("answer") or (streamed if isinstance(streamed, str) else "")
    return turn


def render_chat(collection: dict | None) -> None:
    if collection is None:
        st.info("Create a collection in the sidebar first.")
        return
    name = collection["name"]
    history: list[dict[str, Any]] = st.session_state.setdefault(f"history-{name}", [])
    reidentify = collection["phi_protection"] and st.toggle("Show original identifiers")
    for index, turn in enumerate(history):
        _render_turn(name, turn, f"{name}-{index}")
    question = st.chat_input("Ask about these documents")
    if question:
        try:
            history.append(_ask(name, question, reidentify))
            st.rerun()
        except api_client.ApiError as exc:
            st.error(exc.message)
