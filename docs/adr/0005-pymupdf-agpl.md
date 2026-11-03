# ADR 0005: PyMuPDF for PDF parsing, despite the AGPL

**Status:** Accepted, with a condition. **Date:** 2026-11-01.

## Context

PDFs need per-page text, layout blocks for reading order, page rendering for OCR, and encryption
detection. PyMuPDF does all of these in one library.

## Decision

Use PyMuPDF, imported only in `ingestion/parsers/pdf.py`, and parse in a child process.

## Alternatives

| Option | Why not |
|---|---|
| pypdf | Permissive licence, but no page rendering, so OCR would need a second library |
| pdfplumber with pypdfium2 | Workable replacement, at more code |
| Commercial PyMuPDF licence | A cost to decide on if this is ever sold |

## Consequences

- **Condition:** AGPL can require sharing source with users of a network service. That's fine for a
  public project. A closed commercial deployment needs a legal review, then a licence or a swap.
- A swap touches one module, because everything downstream only sees `Block`s.
