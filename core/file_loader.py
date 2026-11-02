"""
file_loader.py
--------------
Turns user input (pasted text, CSV, TXT or PDF) into a clean list of
complaint strings. All input validation happens here.
"""

from __future__ import annotations

import io

import pandas as pd

MAX_COMPLAINTS = 50        # protects the free-tier rate limit
MAX_CHARS_PER_ITEM = 2000  # very long texts are cut to save tokens
MIN_CHARS = 10             # shorter than this is not a real complaint


class InputError(Exception):
    """Problem with what the user gave us (not with the API)."""


def _clean(items: list[str]) -> list[str]:
    out = []
    for t in items:
        t = " ".join(str(t).split())  # collapse whitespace and newlines
        if len(t) >= MIN_CHARS:
            out.append(t[:MAX_CHARS_PER_ITEM])
    if not out:
        raise InputError(f"No usable complaint found. Each complaint needs at least {MIN_CHARS} characters.")
    if len(out) > MAX_COMPLAINTS:
        out = out[:MAX_COMPLAINTS]
    return out


def from_text(text: str) -> list[str]:
    """One complaint per line (or blocks separated by blank lines)."""
    if not text or not text.strip():
        raise InputError("The text box is empty. Paste at least one complaint.")
    if "\n\n" in text.strip():
        parts = text.strip().split("\n\n")
    else:
        parts = text.strip().splitlines()
    return _clean(parts)


def from_csv(file) -> list[str]:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise InputError(f"Could not read the CSV file: {e}")
    if df.empty:
        raise InputError("The CSV file has no rows.")
    # use a column called complaint/description/text, else the longest text column
    lower = {c.lower(): c for c in df.columns}
    for name in ("complaint", "description", "text", "problem"):
        if name in lower:
            return _clean(df[lower[name]].dropna().tolist())
    text_cols = df.select_dtypes(include="object").columns
    if len(text_cols) == 0:
        raise InputError("No text column found in the CSV. Name it 'complaint'.")
    col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return _clean(df[col].dropna().tolist())


def from_pdf(file) -> list[str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file.read()))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        raise InputError(f"Could not read the PDF: {e}")
    if not text.strip():
        raise InputError("The PDF has no extractable text (it may be a scanned image).")
    return from_text(text)


def load_upload(file) -> list[str]:
    name = file.name.lower()
    if name.endswith(".csv"):
        return from_csv(file)
    if name.endswith(".pdf"):
        return from_pdf(file)
    if name.endswith(".txt"):
        return from_text(file.read().decode("utf-8", errors="ignore"))
    raise InputError("Unsupported file type. Use CSV, TXT or PDF.")
