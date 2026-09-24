from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from config.settings import CHUNK_SIZE, CORPUS_NAME, DATA_PATH, DOC_VERSION


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"^#+\s*", "", text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "section")


def _heading_from_section(section: str) -> tuple[str, int]:
    first = section.split("\n", 1)[0].strip()
    match = re.match(r"^(#{2,3})\s+(.*)", first)
    if not match:
        return "preamble", 0
    return match.group(2).strip(), len(match.group(1))


def _pack_section(section: str, size: int) -> list[str]:
    if len(section) <= size:
        return [section]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
    parts: list[str] = []
    current = ""
    prev_para = ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                parts.append(current)
                current = ""
            for i in range(0, len(para), size):
                parts.append(para[i : i + size])
            prev_para = para
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                parts.append(current)
            overlapped = f"{prev_para}\n\n{para}" if prev_para else para
            current = overlapped if len(overlapped) <= size else para
        prev_para = para

    if current:
        parts.append(current)
    return parts


def chunk_text(
    path: str | Path | None = None,
    chunk_size: int | None = None,
) -> list[dict]:
    """Return chunks with stable ids + metadata for one Chroma collection."""
    source = Path(path) if path is not None else DATA_PATH
    size = chunk_size if chunk_size is not None else CHUNK_SIZE
    content = source.read_text(encoding="utf-8")
    date_added = datetime.now(timezone.utc).date().isoformat()

    sections = re.split(r"(?=^#{2,3} )", content, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[dict] = []

    for section_index, section in enumerate(sections):
        heading, heading_level = _heading_from_section(section)
        heading_slug = slugify(heading)
        parts = _pack_section(section, size)

        for part_index, text in enumerate(parts):
            chunk_id = f"{CORPUS_NAME}-{heading_slug}-{section_index}-{part_index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "document": text,
                    "metadata": {
                        "corpus": CORPUS_NAME,
                        "source_file": source.name,
                        "heading": heading,
                        "heading_level": heading_level,
                        "section": heading_slug,
                        "section_index": section_index,
                        "part_index": part_index,
                        "doc_version": DOC_VERSION,
                        "date_added": date_added,
                    },
                }
            )

    return chunks
