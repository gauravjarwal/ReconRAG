"""A tripwire for prompt injection, not the defence.

Regexes are easy to bypass (other languages, encodings). The real defence is that retrieved text
is delimited, marked as data, and the model has no tools to misuse. A flag here only adds a warning.
"""

from __future__ import annotations

import re

_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"override\s+(previous|prior|all)\s+(instructions?|rules?)",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"do\s+not\s+follow\s+(previous|prior|your)",
        r"you\s+are\s+now\s+",
        r"new\s+instructions?\s*:",
        r"\bsystem\s*:\s*",
        r"###\s*(system|instruction)",
        r"<\|(im_start|system)\|>",
        r"\[INST\]",
        r"reveal\s+(your\s+)?(system\s+)?prompt",
    )
]
_DELIMITERS = re.compile(r"<(/?)(source|sources|question)\b", re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PATTERNS)


def escape_delimiters(text: str) -> str:
    """Untrusted text can't close our prompt blocks early."""
    return _DELIMITERS.sub(r"&lt;\1\2", text)
