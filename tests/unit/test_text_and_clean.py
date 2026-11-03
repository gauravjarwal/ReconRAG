from reconrag.domain import Block, BlockKind, ParsedDocument
from reconrag.ingestion.clean import clean_document
from reconrag.ingestion.quality import garble_ratio, score
from reconrag.ingestion.text import clean_text, decode


def text_block(text: str, page: int | None = None) -> Block:
    return Block(BlockKind.TEXT, text, page=page)


def test_decode_handles_common_encodings():
    assert decode("café".encode()) == "café"
    assert decode("café".encode("utf-16")) == "café"
    assert decode("﻿hello".encode()) == "hello"
    assert "caf" in decode("le café est très bon, merci beaucoup".encode("latin-1"))


def test_mojibake_is_repaired():
    assert clean_text("cafÃ©") == "café"


def test_invisible_characters_are_removed():
    assert clean_text("ab​c­de\x07f") == "abcdef"


def test_words_broken_at_line_end_rejoin_but_real_hyphens_stay():
    assert clean_text("insur-\nance") == "insurance"
    assert clean_text("well-known") == "well-known"
    assert clean_text("A-\nB") == "A-\nB"


def test_spacing_is_normalised():
    assert clean_text("a  \t b c\n\n\n\nd") == "a b c\n\nd"


def test_repeated_header_and_footer_go_but_body_stays():
    blocks: list[Block] = []
    for page in range(1, 6):
        blocks += [
            text_block("Tarn Ltd Confidential", page),
            text_block(
                f"Body paragraph {page} is unique to this page and mentions {page * 7}.", page
            ),
            text_block(f"Page {page} of 5", page),
        ]
    cleaned = clean_document(ParsedDocument(blocks, parser="t"))
    texts = [block.text for block in cleaned.blocks]
    assert len(texts) == 5
    assert all(text.startswith("Body paragraph") for text in texts)


def test_short_documents_keep_their_repeated_lines():
    blocks = [text_block("Same line", 1), text_block("Same line", 2)]
    assert len(clean_document(ParsedDocument(blocks, parser="t")).blocks) == 2


def test_table_cells_are_cleaned_and_empty_rows_dropped():
    block = Block(BlockKind.TABLE, rows=[["a​", "b"], ["", " "]])
    assert clean_document(ParsedDocument([block], parser="t")).blocks[0].rows == [["a", "b"]]


def test_garbage_scores_below_clean_text():
    good, _ = score(
        ParsedDocument([text_block("A normal sentence about claims. " * 10)], parser="t")
    )
    bad, _ = score(ParsedDocument([text_block("�#~^ §¶ ¤¤ ¥¦ ~~ ^^ " * 10)], parser="t"))
    assert good > 0.9
    assert bad < 0.2


def test_ocr_confidence_lowers_the_score_and_is_flagged():
    document = ParsedDocument(
        [text_block("A normal sentence about claims. " * 10)], parser="t", ocr_confidence=0.5
    )
    value, flags = score(document)
    assert value < 0.6
    assert "ocr" in flags


def test_garble_ratio_of_empty_text_is_zero():
    assert garble_ratio("") == 0.0
