import io
import zipfile

import pytest

from reconrag.config import Settings
from reconrag.errors import RejectedFile
from reconrag.ingestion.sniff import check_zip, sniff


def zipped(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("data", "name", "kind"),
    [
        (b"%PDF-1.7 rest", "a.pdf", "pdf"),
        (b"%PDF-1.7 rest", "misnamed.txt", "pdf"),
        (b"\x89PNG\r\n\x1a\nxxx", "a.png", "image"),
        (b"<!doctype html><html></html>", "page.txt", "html"),
        (b"plain text", "notes.txt", "txt"),
        (b"# head", "a.md", "md"),
        (b"a,b\n1,2", "a.csv", "csv"),
        ("hello".encode("utf-16"), "a.txt", "txt"),
    ],
)
def test_content_decides_the_type(settings: Settings, data: bytes, name: str, kind: str):
    assert sniff(data, name, settings) == kind


def test_office_files_are_told_apart_by_their_contents(settings: Settings):
    assert sniff(zipped({"word/document.xml": b"x"}), "a.docx", settings) == "docx"
    assert sniff(zipped({"xl/workbook.xml": b"x"}), "renamed.docx", settings) == "xlsx"


@pytest.mark.parametrize(
    ("data", "name", "code"),
    [
        (b"", "a.txt", "empty_file"),
        (b"   \n ", "a.txt", "empty_file"),
        (b"MZ\x90\x00binary", "invoice.pdf", "unsupported_type"),
        (b"\xd0\xcf\x11\xe0legacy", "old.doc", "unsupported_type"),
        (b"just text", "fake.pdf", "type_mismatch"),
        (b"some text", "script.py", "unsupported_type"),
        (b"\x00\x01\x02\x03" * 100, "blob.txt", "unsupported_type"),
        (b"PK\x03\x04broken", "a.docx", "corrupt_file"),
    ],
)
def test_bad_files_get_a_stable_code(settings: Settings, data: bytes, name: str, code: str):
    with pytest.raises(RejectedFile) as caught:
        sniff(data, name, settings)
    assert caught.value.code == code


def test_plain_zip_is_refused(settings: Settings):
    with pytest.raises(RejectedFile) as caught:
        sniff(zipped({"a.txt": b"x"}), "a.docx", settings)
    assert caught.value.code == "unsupported_type"


def test_zip_bomb_is_refused_from_the_directory_alone():
    settings = Settings(_env_file=None, max_unzipped_bytes=1000)  # type: ignore[call-arg]
    with pytest.raises(RejectedFile) as caught:
        check_zip(zipped({"word/document.xml": b"0" * 2000}), settings)
    assert caught.value.code == "zip_bomb"


def test_highly_compressed_large_file_is_refused():
    settings = Settings(_env_file=None, max_zip_ratio=10)  # type: ignore[call-arg]
    with pytest.raises(RejectedFile) as caught:
        check_zip(zipped({"word/document.xml": b"0" * (11 * 1024 * 1024)}), settings)
    assert caught.value.code == "zip_bomb"
