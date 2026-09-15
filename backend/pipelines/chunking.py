CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
MIN_CHUNK_LENGTH = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _merge_pieces(pieces: list[str], separator: str, chunk_size: int, overlap: int) -> list[str]:
    """Merge a list of text pieces into chunks respecting size and overlap."""
    chunks: list[str] = []
    current = ""

    for part in pieces:
        piece = part + separator if separator else part
        if len(current) + len(piece) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
            if len(current) > overlap:
                current = current[-overlap:] + piece
            else:
                current = piece
        else:
            current += piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Split text on the first separator that exists in it, then recursively
    split any oversized pieces on the next separator down the list.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for i, sep in enumerate(separators):
        if sep and sep in text:
            parts = text.split(sep)
            merged = _merge_pieces(parts, sep, chunk_size, overlap)

            # Recurse on chunks still too large using remaining separators
            remaining_seps = separators[i + 1:]
            result: list[str] = []
            for chunk in merged:
                if len(chunk) > chunk_size and remaining_seps:
                    result.extend(_recursive_split(chunk, remaining_seps, chunk_size, overlap))
                else:
                    result.append(chunk)
            return result

    # No separator found — hard split by characters
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks using recursive separator splitting.
    Splits on \\n\\n first, then recursively splits oversized pieces on \\n, then '. ', etc.
    Filters out chunks shorter than MIN_CHUNK_LENGTH.
    """
    if not text.strip():
        return []

    chunks = _recursive_split(text, SEPARATORS, chunk_size, overlap)

    # Hard-split any chunks still over the limit (e.g. long words, no separators)
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            result.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                result.append(chunk[start:start + chunk_size])
                start += chunk_size - overlap

    return [c for c in result if len(c.strip()) >= MIN_CHUNK_LENGTH]
