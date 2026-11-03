"""Turns world facts into messy files. Each helper reproduces one real-world failure on purpose."""

from __future__ import annotations

import csv
import io
import os
import random
import re
import textwrap
import zipfile
from collections.abc import Sequence
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import docx
import numpy as np
import openpyxl
import reportlab
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors, pdfencrypt
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from evals.corpus.world import (
    DEFINITIONS,
    GENERAL_CONDITIONS,
    INSURER,
    NOTICE,
    Clause,
    Plan,
    indian_format,
)

FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_VERA = os.path.join(os.path.dirname(reportlab.__file__), "fonts", "Vera.ttf")
_STYLES = getSampleStyleSheet()
_BODY = ParagraphStyle(
    "Body", parent=_STYLES["BodyText"], fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY
)
_HEAD = ParagraphStyle("Head", parent=_STYLES["Heading3"], fontSize=11, spaceBefore=8)
_TITLE = ParagraphStyle("TitleX", parent=_STYLES["Title"], fontSize=16)


def fact_sentences(plan: Plan) -> dict[str, str]:
    """The plan-specific sentences. Golden quotes reuse these exact strings."""
    return {
        "excess": f"An excess of Rs. {indian_format(plan.excess)} applies to each claim under "
        "this plan.",
        "sum_insured": f"The sum insured for this plan is Rs. {indian_format(plan.sum_insured)} "
        "in any policy year.",
        "waiting": f"No claim is payable for an event in the first {plan.waiting_days} days "
        "after the start date.",
        "free_look": f"You may return this policy within {plan.free_look_days} days of receipt "
        "for a refund, less the deductions allowed by the regulator.",
        "claim_window": f"Every claim must be notified to us within {plan.claim_window_days} "
        "days of the event.",
        "grace": f"Renewal premium may be paid up to {plan.grace_days} days after expiry without "
        "loss of continuity.",
        "max_claims": f"No more than {plan.max_claims} claims are payable in a policy year.",
    }


def policy_sections(plan: Plan, clauses: Sequence[Clause]) -> list[tuple[str, list[str]]]:
    facts = fact_sentences(plan)
    sections: list[tuple[str, list[str]]] = [
        ("1. Definitions", [f"{term}: {meaning}" for term, meaning in DEFINITIONS]),
        (
            "2. What this policy covers",
            [
                f"This {plan.name} policy covers {plan.line.covers} against the events "
                "described in "
                "this wording, up to the limits in the schedule.",
                facts["sum_insured"],
                facts["excess"],
                facts["max_claims"],
            ],
        ),
    ]
    sections.extend(
        (f"{i}. {clause.heading}", [clause.text]) for i, clause in enumerate(clauses, start=3)
    )
    next_no = len(clauses) + 3
    sections.append(
        (f"{next_no}. Waiting period and claims", [facts["waiting"], facts["claim_window"]])
    )
    sections.append(
        (f"{next_no + 1}. Cancellation and renewal", [facts["free_look"], facts["grace"]])
    )
    sections.append(
        (
            f"{next_no + 2}. General conditions",
            [f"{title}. {text}" for title, text in GENERAL_CONDITIONS],
        )
    )
    return sections


def plain_text(sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    return "\n\n".join(f"{head}\n" + "\n".join(paragraphs) for head, paragraphs in sections)


def _break_words(text: str, rng: random.Random) -> str:
    """Force mid-word line breaks, the way justified PDF layouts hyphenate."""

    def split_word(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word) < 9 or rng.random() > 0.12:
            return word
        cut = len(word) // 2
        return f"{word[:cut]}-<br/>{word[cut:]}"

    return re.sub(r"[A-Za-z]+", split_word, escape(text))


def _page_furniture(header: str, footer: str):
    def draw(canvas: Canvas, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, A4[1] - 1.3 * cm, header)
        canvas.drawString(2 * cm, 1.2 * cm, footer)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    return draw


def _schedule_table(plan: Plan) -> Table:
    rows = [
        ["Item", "Detail"],
        ["Plan code", plan.code],
        ["Edition", str(plan.edition)],
        ["Annual premium", f"Rs. {indian_format(plan.premium)}"],
        ["Sum insured", f"Rs. {indian_format(plan.sum_insured)}"],
        ["Excess", f"Rs. {indian_format(plan.excess)}"],
    ]
    table = Table(rows, colWidths=[3.2 * cm, 3.8 * cm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]
        )
    )
    return table


def write_policy_pdf(
    path: Path,
    plan: Plan,
    clauses: Sequence[Clause],
    columns: int,
    hyphenate: bool,
    rng: random.Random,
) -> None:
    """A native policy wording with running headers, one or two columns and broken words."""
    header = f"{INSURER} | {plan.name} Policy Wording | Edition {plan.edition}"
    footer = f"{NOTICE} Form TA/{plan.line.code}/{plan.edition}"
    width, height = A4
    margin = 2 * cm
    if columns == 2:
        gap = 0.6 * cm
        col = (width - 2 * margin - gap) / 2
        frames = [
            Frame(margin, margin, col, height - 2 * margin, id="left"),
            Frame(margin + col + gap, margin, col, height - 2 * margin, id="right"),
        ]
    else:
        frames = [Frame(margin, margin, width - 2 * margin, height - 2 * margin, id="main")]
    document = BaseDocTemplate(
        str(path),
        pagesize=A4,
        invariant=1,
        title=f"{plan.name} {plan.edition}",
        pageTemplates=[PageTemplate(id="p", frames=frames, onPage=_page_furniture(header, footer))],
    )
    story: list[Any] = [
        Paragraph(f"{plan.name} Policy Wording", _TITLE),
        Spacer(1, 6),
        _schedule_table(plan),
        Spacer(1, 8),
    ]
    for head, paragraphs in policy_sections(plan, clauses):
        story.append(Paragraph(escape(head), _HEAD))
        for paragraph in paragraphs:
            text = _break_words(paragraph, rng) if hyphenate else escape(paragraph)
            story.append(Paragraph(text, _BODY))
    document.build(story)


def write_interleaved_pdf(path: Path, plan: Plan, clauses: Sequence[Clause]) -> None:
    """Two visual columns whose text objects run across the page row by row.

    A human reads each column down. A naive extractor reads left line, right line, left line.
    This is the classic multi-column reading-order failure.
    """
    header = f"{INSURER} | {plan.name} Policy Wording | Edition {plan.edition}"
    footer = f"{NOTICE} Form TA/{plan.line.code}/{plan.edition}"
    width, height = A4
    margin = 2 * cm
    gap = 0.8 * cm
    col = (width - 2 * margin - gap) / 2
    leading = 11.5
    lines_per_col = int((height - 2 * margin - 1.5 * cm) / leading)

    lines: list[tuple[str, str]] = [
        ("Helvetica-Bold", f"{plan.name} Policy Wording"),
        ("Helvetica", f"Plan code {plan.code}. Annual premium Rs. {indian_format(plan.premium)}."),
    ]
    for head, paragraphs in policy_sections(plan, clauses):
        lines.append(("Helvetica-Bold", head))
        for paragraph in paragraphs:
            lines.extend(("Helvetica", row) for row in simpleSplit(paragraph, "Helvetica", 9, col))

    canvas = Canvas(str(path), pagesize=A4, invariant=1)
    pages = [lines[i : i + 2 * lines_per_col] for i in range(0, len(lines), 2 * lines_per_col)]
    for number, page in enumerate(pages, start=1):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(margin, height - 1.3 * cm, header)
        canvas.drawString(margin, 1.2 * cm, footer)
        canvas.drawRightString(width - margin, 1.2 * cm, f"Page {number}")
        left, right = page[:lines_per_col], page[lines_per_col:]
        for row in range(lines_per_col):
            y = height - margin - 0.8 * cm - row * leading
            if row < len(left):
                canvas.setFont(left[row][0], 9)
                canvas.drawString(margin, y, left[row][1])
            if row < len(right):
                canvas.setFont(right[row][0], 9)
                canvas.drawString(margin + col + gap, y, right[row][1])
        canvas.showPage()
    canvas.save()


def write_borderless_table_pdf(
    path: Path,
    title: str,
    intro: str,
    column_names: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    """A table with no rules at all, so structure exists only as whitespace."""
    document = BaseDocTemplate(
        str(path),
        pagesize=A4,
        invariant=1,
        title=title,
        pageTemplates=[
            PageTemplate(
                id="p",
                frames=[Frame(2 * cm, 2 * cm, A4[0] - 4 * cm, A4[1] - 4 * cm, id="main")],
                onPage=_page_furniture(f"{INSURER} | Service Summary", NOTICE),
            )
        ],
    )
    table = Table([list(column_names), *[list(row) for row in rows]])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    document.build(
        [
            Paragraph(escape(title), _TITLE),
            Spacer(1, 6),
            Paragraph(escape(intro), _BODY),
            Spacer(1, 10),
            table,
        ]
    )


def _scan_image(
    lines: Sequence[str], rng: random.Random, np_rng: np.random.Generator
) -> Image.Image:
    """Render text, then degrade it the way a tired office scanner would."""
    width, height = 1240, 1754
    page = Image.new("L", (width, height), 245)
    draw = ImageDraw.Draw(page)
    body = ImageFont.truetype(_VERA, 30)
    title = ImageFont.truetype(_VERA, 40)
    y = 150
    for i, line in enumerate(lines):
        font = title if i == 0 else body
        for row in textwrap.wrap(line, 58) or [""]:
            draw.text((120, y), row, fill=25, font=font)
            y += 52
        y += 24
    page = page.rotate(rng.uniform(-1.6, 1.6), fillcolor=245, resample=Image.Resampling.BICUBIC)
    page = page.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.0)))
    pixels = np.asarray(page, dtype=np.int16)
    pixels = pixels + np_rng.normal(0, 7, pixels.shape).astype(np.int16)
    speckle = np_rng.random(pixels.shape) < 0.0008
    pixels[speckle] = 30
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=55)
    return buffer.getvalue()


def write_scanned_pdf(
    path: Path, lines: Sequence[str], rng: random.Random, np_rng: np.random.Generator
) -> None:
    """An image-only PDF with no text layer at all."""
    canvas = Canvas(str(path), pagesize=A4, invariant=1)
    image = ImageReader(io.BytesIO(_jpeg(_scan_image(lines, rng, np_rng))))
    canvas.drawImage(image, 0, 0, width=A4[0], height=A4[1])
    canvas.showPage()
    canvas.save()


def write_mixed_pdf(
    path: Path,
    native_lines: Sequence[str],
    scanned_lines: Sequence[str],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> None:
    """Page one has a text layer, page two is a scan. Only per-page routing reads both."""
    canvas = Canvas(str(path), pagesize=A4, invariant=1)
    y = A4[1] - 3 * cm
    for line in native_lines:
        canvas.setFont("Helvetica", 11)
        for row in simpleSplit(line, "Helvetica", 11, A4[0] - 4 * cm):
            canvas.drawString(2 * cm, y, row)
            y -= 15
        y -= 8
    canvas.showPage()
    image = ImageReader(io.BytesIO(_jpeg(_scan_image(scanned_lines, rng, np_rng))))
    canvas.drawImage(image, 0, 0, width=A4[0], height=A4[1])
    canvas.showPage()
    canvas.save()


def write_scanned_png(
    path: Path, lines: Sequence[str], rng: random.Random, np_rng: np.random.Generator
) -> None:
    _scan_image(lines, rng, np_rng).resize((930, 1315)).save(path, format="PNG")


def write_blank_pdf(path: Path) -> None:
    canvas = Canvas(str(path), pagesize=A4, invariant=1)
    for _ in range(3):
        canvas.showPage()
    canvas.save()


def write_encrypted_pdf(path: Path, text: str) -> None:
    encryption = pdfencrypt.StandardEncryption("not-the-password", canPrint=0, strength=128)
    canvas = Canvas(str(path), pagesize=A4, invariant=1, encrypt=encryption)
    canvas.drawString(2 * cm, A4[1] - 3 * cm, text)
    canvas.showPage()
    canvas.save()


def write_hidden_text_pdf(path: Path, visible: Sequence[str], hidden: str) -> None:
    """White text a reader never sees, but a parser extracts. A standard injection route."""
    canvas = Canvas(str(path), pagesize=A4, invariant=1)
    y = A4[1] - 3 * cm
    for line in visible:
        canvas.setFont("Helvetica", 11)
        for row in simpleSplit(line, "Helvetica", 11, A4[0] - 4 * cm):
            canvas.drawString(2 * cm, y, row)
            y -= 15
        y -= 8
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 2)
    canvas.drawString(2 * cm, 3 * cm, hidden)
    canvas.showPage()
    canvas.save()


def write_docx(
    path: Path,
    title: str,
    paragraphs: Sequence[tuple[str, str]],
    table: Sequence[Sequence[str]] | None,
    footer: str,
    real_headings: bool,
) -> None:
    """Paragraphs are (kind, text). Half the files fake headings in bold, as people really do."""
    document = docx.Document()
    if real_headings:
        document.add_heading(title, level=1)
    else:
        document.add_paragraph().add_run(title).bold = True
    for kind, text in paragraphs:
        if kind == "heading":
            if real_headings:
                document.add_heading(text, level=2)
            else:
                document.add_paragraph().add_run(text.upper()).bold = True
        elif kind == "step":
            document.add_paragraph(text, style="List Number")
        else:
            document.add_paragraph(text)
    if table:
        grid = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                grid.cell(r, c).text = cell
    document.sections[0].footer.paragraphs[0].text = footer
    document.save(str(path))
    normalize_office_zip(path)


def write_xlsx(path: Path, sheets: dict[str, Sequence[Sequence[Any]]]) -> None:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(list(row))
    workbook.save(path)
    normalize_office_zip(path)


def normalize_office_zip(path: Path) -> None:
    """Office files embed timestamps. Pin them so a rebuild is byte-identical."""
    with zipfile.ZipFile(path) as source:
        entries = [(name, source.read(name)) for name in sorted(source.namelist())]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in entries:
            if name == "docProps/core.xml":
                data = re.sub(
                    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", b"2026-01-01T00:00:00Z", data
                )
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, data)


def write_csv(path: Path, rows: Sequence[Sequence[Any]], delimiter: str, encoding: str) -> None:
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n").writerows(rows)
    path.write_bytes(buffer.getvalue().encode(encoding))


def html_page(title: str, body_html: str, hidden: str = "") -> str:
    """A real page: most bytes are navigation, banners and scripts."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title} | {INSURER}</title>
<style>.nav{{display:flex}} .banner{{position:fixed}}</style>
<script>window.dataLayer=[];function track(e){{dataLayer.push(e)}}track('pageview');</script>
</head><body>
<nav class="nav"><a href="/">Home</a><a href="/plans">Our plans</a>\
<a href="/claims">Make a claim</a>
<a href="/contact">Contact us</a><a href="/careers">Careers at {INSURER}</a></nav>
<div class="banner" id="cookies">We use cookies to improve your experience.
<button>Accept all cookies</button></div>
<main><h1>{title}</h1>
{body_html}
{hidden}
</main>
<aside><h2>You may also like</h2><ul><li>Refer a friend and earn rewards</li>
<li>Download the {INSURER} mobile app</li></ul></aside>
<footer>Copyright 2026 {INSURER}. All rights reserved. {NOTICE}
<a href="/privacy">Privacy policy</a> <a href="/terms">Terms of use</a></footer>
</body></html>
"""


def write_email(
    path: Path,
    subject: str,
    sender: str,
    recipient: str,
    date: str,
    body: str,
    html_body: str | None,
    charset: str,
    boundary: str,
) -> None:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message["Date"] = date
    message["Message-ID"] = f"<{boundary}@mail.tarnbrook.example>"
    message.set_content(body, charset=charset)
    if html_body is not None:
        message.add_alternative(html_body, subtype="html", charset=charset)
        message.set_boundary(boundary)
    path.write_bytes(message.as_bytes())


def mojibake(text: str) -> str:
    """UTF-8 bytes read as cp1252: the classic â€™ damage."""
    return text.encode("utf-8").decode("cp1252", errors="replace")
