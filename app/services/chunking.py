"""Split page texts into overlapping chunks that keep their page number."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkDraft:
    page: int  # 1-based
    position: int  # order in the whole document
    text: str


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[ChunkDraft]:
    """Cut each page into windows of about `size` characters.

    Windows end on a word boundary when possible, and consecutive windows share about
    `overlap` characters so that a sentence cut in two still appears whole in one chunk.
    Chunks never span two pages, which keeps page citations exact.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")

    drafts: list[ChunkDraft] = []
    for page_number, raw in enumerate(pages, start=1):
        text = " ".join(raw.split())  # normalise whitespace and line breaks
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # Move the cut back to the last space so we don't split a word.
                space = text.rfind(" ", start + size // 2, end)
                if space != -1:
                    end = space
            piece = text[start:end].strip()
            if piece:
                drafts.append(ChunkDraft(page=page_number, position=len(drafts), text=piece))
            if end >= len(text):
                break
            next_start = end - overlap
            # Start the next window on a word boundary too.
            space = text.find(" ", next_start, end)
            start = space + 1 if space != -1 else next_start
    return drafts
