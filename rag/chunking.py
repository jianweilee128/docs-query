import re
from pathlib import Path

from config.settings import CHUNK_SIZE, DATA_PATH


def chunk_text(
    path: str | Path | None = None,
    chunk_size: int | None = None,
) -> list[str]:
    source = Path(path) if path is not None else DATA_PATH
    size = chunk_size if chunk_size is not None else CHUNK_SIZE
    content = source.read_text(encoding="utf-8")

    # 1) Split on ## / ### headers (keep the heading on each section)
    sections = re.split(r"(?=^#{2,3} )", content, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[str] = []

    for section in sections:
        if len(section) <= size:
            chunks.append(section)
            continue

        # 2) Pack paragraphs; on overflow, start next chunk with prev paragraph overlap
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        current = ""
        prev_para = ""

        for para in paragraphs:
            # Single paragraph still too big → hard-split (last resort)
            if len(para) > size:
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(para), size):
                    chunks.append(para[i : i + size])
                prev_para = para
                continue

            candidate = f"{current}\n\n{para}" if current else para
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # 3) Overlap: carry previous paragraph into the new chunk
                overlapped = f"{prev_para}\n\n{para}" if prev_para else para
                current = overlapped if len(overlapped) <= size else para

            prev_para = para

        if current:
            chunks.append(current)

    return chunks
