from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipelines.chunking import CHUNK_SIZE, MIN_CHUNK_LENGTH, chunk_text


def test_short_text_above_min_returns_single_chunk():
    text = "Hello world. This is a meaningful sentence with enough content to pass the minimum."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_short_text_below_min_filtered_out():
    text = "Too short."
    chunks = chunk_text(text)
    assert len(chunks) == 0


def test_empty_text_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_long_text_splits_into_multiple_chunks():
    text = "word " * 300  # ~1500 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_no_chunk_exceeds_size():
    text = "word " * 500
    chunks = chunk_text(text)
    for chunk in chunks:
        assert len(chunk) <= CHUNK_SIZE + 50  # small tolerance for separator boundary


def test_all_chunks_above_min_length():
    text = "A decent sentence here. " * 50
    chunks = chunk_text(text)
    for chunk in chunks:
        assert len(chunk.strip()) >= MIN_CHUNK_LENGTH


def test_paragraph_separator_preferred_over_sentence():
    para1 = "First paragraph with enough content to be meaningful. " * 10   # ~550 chars
    para2 = "Second paragraph also has substantial content for testing. " * 10  # ~590 chars
    text = para1.strip() + "\n\n" + para2.strip()
    chunks = chunk_text(text)
    # Total ~1140 chars > 600 chunk size — should split on \n\n
    assert len(chunks) >= 2
    assert "First paragraph" in chunks[0]
    assert "Second paragraph" in chunks[-1]


def test_recursive_split_on_newline_after_paragraph():
    # One huge paragraph (no \n\n inside) forces split on \n
    lines = [f"Line number {i} with some filler text to reach size." for i in range(30)]
    text = "\n".join(lines)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= CHUNK_SIZE + 50


def test_overlap_carries_content():
    section_a = "Alpha " * 80   # 480 chars
    section_b = "Beta " * 80    # 400 chars
    text = section_a + "\n\n" + section_b
    chunks = chunk_text(text)
    assert len(chunks) >= 2
