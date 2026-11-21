"""Parse 3GPP .docx specifications into structural chunks with metadata.

Usage:
    python -m ingest.parse_docx                      # all corpus/raw/*.docx -> corpus/chunks.jsonl
    python -m ingest.parse_docx FILE.docx -o out.jsonl

Pipeline position: corpus (.docx) -> [THIS] -> chunks.jsonl -> embeddings / BM25.

What it does:
  1. Walks the document body in order (paragraphs and tables).
  2. Detects section headings from Word styles and splits "5.5.1.2<TAB>Title" into number + title.
  3. Gathers the text under each heading until the next heading (any level).
  4. Converts tables to Markdown; keeps table/figure captions; drops images.
  5. Skips noise: cover page, table of contents, Foreword, Void sections, Change history.
  6. Splits long sections into overlapping sub-chunks (same section metadata).
  7. Prefixes every chunk with its spec + section breadcrumb (helps retrieval).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "corpus" / "raw"
OUT_FILE = ROOT / "corpus" / "chunks.jsonl"

# Chunk size limits, in approximate tokens (1 token ~ 4 characters of English text).
MAX_TOKENS = 400  # bge-base-en-v1.5 reads at most 512 tokens
OVERLAP_TOKENS = 60
CHARS_PER_TOKEN = 4

# Word style -> heading level. Annexes (Heading 8) are top level, like clauses.
HEADING_LEVELS = {f"Heading {i}": i for i in range(1, 8)}
HEADING_LEVELS.update({"Heading 8": 1, "H2": 2, "H6": 6})

SKIP_TITLES = re.compile(r"^(foreword|void|change history)$", re.I)
SKIP_STYLES = ("toc", "TT")  # table of contents entries and "Contents" title
LIST_STYLES = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4}


def approx_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\t", " ")
    return re.sub(r"[ ]{2,}", " ", text).strip()


def spec_metadata(path: Path) -> dict:
    """Decode 3GPP file names: '24501-ie0' -> TS 24.501 V18.14.0 (letters: a=10, b=11, ...)."""
    m = re.match(r"(\d{2})(\d{3})-([0-9a-z])([0-9a-z])([0-9a-z])", path.stem)
    if not m:
        raise ValueError(f"Unexpected 3GPP file name: {path.name}")
    series, num, *ver = m.groups()
    major, minor, patch = (int(c, 36) for c in ver)
    return {
        "spec": f"TS {series}.{num}",
        "version": f"{major}.{minor}.{patch}",
        "release": major,
        "source_file": path.name,
    }


def parse_heading(text: str, style: str) -> tuple[str, str]:
    """Return (section_number, title) for a heading paragraph."""
    text = text.replace("\xa0", " ").strip()
    if style == "Heading 8" or text.lower().startswith("annex"):
        # "Annex A (informative):\nCause values for 5GS mobility management"
        m = re.match(r"(Annex\s+[A-Z0-9]+)\s*(\([^)]*\))?\s*:?\s*(.*)", text, re.S)
        if m:
            return m.group(1), clean(m.group(3)) or m.group(1)
    # "5.5.1.2", "4.2.8.1A", "5.20b.1", "A.1", "D.2.3"
    m = re.match(r"^((?:[A-Z]\.)?\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*)[\s\t]+(.*)", text, re.S)
    if m:
        return m.group(1), clean(m.group(2))
    return "", clean(text)  # unnumbered (e.g. "Foreword")


def heading_level(number: str, style: str) -> int:
    """Level from the clause number (3GPP styles annex 'A.1' as Heading 1, so the style is unreliable)."""
    if number.startswith("Annex"):
        return 1
    if number:
        return number.count(".") + 1
    return HEADING_LEVELS[style]


def table_to_markdown(table: Table) -> str:
    """Render a table as Markdown. Reads <w:tc> directly so merged cells are not duplicated."""
    rows = []
    for tr in table._tbl.iter(qn("w:tr")):
        cells = []
        for tc in tr.iter(qn("w:tc")):
            paras = [clean("".join(t.text or "" for t in p.iter(qn("w:t"))))
                     for p in tc.iter(qn("w:p"))]
            cells.append(" <br> ".join(p for p in paras if p).replace("|", "\\|"))
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


@dataclass
class Section:
    number: str
    title: str
    level: int
    path: list[tuple[str, str]]  # breadcrumb [(number, title), ...] including self
    blocks: list[str] = field(default_factory=list)
    skip: bool = False


def iter_sections(document: docx.document.Document):
    """Yield Section objects in document order."""
    stack: list[tuple[int, str, str, bool]] = []  # (level, number, title, skip)
    current: Section | None = None

    for el in document.element.body.iterchildren():
        if el.tag == qn("w:tbl"):
            if current is not None and not current.skip:
                md = table_to_markdown(Table(el, document))
                if md:
                    current.blocks.append(md)
            continue
        if el.tag != qn("w:p"):
            continue

        p = Paragraph(el, document)
        style = p.style.name if p.style is not None else ""
        if style.startswith(SKIP_STYLES):
            continue

        if style in HEADING_LEVELS and p.text.strip():
            if current is not None:
                yield current
            number, title = parse_heading(p.text, style)
            level = heading_level(number, style)
            while stack and stack[-1][0] >= level:
                stack.pop()
            skip = bool(SKIP_TITLES.match(title)) or any(s[3] for s in stack)
            stack.append((level, number, title, skip))
            current = Section(number, title, level, [(s[1], s[2]) for s in stack], skip=skip)
            continue

        if current is None or current.skip:
            continue  # cover page, or inside a skipped section
        text = clean(p.text)
        if not text:
            continue  # empty line or image-only paragraph (figures are dropped)
        if style == "TF":
            text = f"[{text}]"  # keep figure caption as a marker
        elif style == "TH":
            text = f"**{text}**"  # table title, followed by the table itself
        elif style in LIST_STYLES:
            text = "  " * LIST_STYLES[style] + text
        current.blocks.append(text)

    if current is not None:
        yield current


def split_blocks(blocks: list[str], budget_chars: int, overlap_chars: int) -> list[str]:
    """Group blocks into pieces under budget, with overlap; oversized tables split by rows."""
    units: list[str] = []
    for b in blocks:
        if len(b) <= budget_chars:
            units.append(b)
        elif b.startswith("|"):  # large table: repeat header on every piece
            lines = b.split("\n")
            header, body, rows = lines[:2], lines[2:], []
            if len(lines[0]) > budget_chars // 4:  # first row is content, not a header
                header, body = [], [lines[0]] + lines[2:]
            for row in body:
                if len("\n".join(header + [row])) <= budget_chars:
                    rows.append(row)
                    continue
                # a single row too large (whole IE description in one cell): split at <br>
                buf = ""
                for part in row.split(" <br> "):
                    if buf and len("\n".join(header)) + len(buf) + len(part) > budget_chars:
                        rows.append(buf)
                        buf = ""
                    buf = f"{buf} <br> {part}" if buf else part
                if buf:
                    rows.append(buf)
            piece: list[str] = []
            for row in rows:
                if piece and len("\n".join(header + piece + [row])) > budget_chars:
                    units.append("\n".join(header + piece))
                    piece = []
                piece.append(row)
            if piece:
                units.append("\n".join(header + piece))
        else:  # very long paragraph: split on sentences
            sentences = re.split(r"(?<=[.;:])\s+", b)
            buf = ""
            for s in sentences:
                if buf and len(buf) + len(s) > budget_chars:
                    units.append(buf)
                    buf = ""
                buf = f"{buf} {s}".strip()
            if buf:
                units.append(buf)

    pieces: list[str] = []
    cur: list[str] = []
    for u in units:
        if cur and len("\n".join(cur + [u])) > budget_chars:
            pieces.append("\n".join(cur))
            # overlap: carry trailing blocks of the previous piece (not tables)
            tail: list[str] = []
            for prev in reversed(cur):
                if prev.startswith("|") or len("\n".join(tail + [prev])) > overlap_chars:
                    break
                tail.insert(0, prev)
            cur = tail
        cur.append(u)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def chunk_document(path: Path) -> list[dict]:
    meta = spec_metadata(path)
    document = docx.Document(str(path))
    chunks: list[dict] = []
    seen: dict[str, int] = {}  # the spec itself sometimes reuses a clause number
    for sec in iter_sections(document):
        if sec.skip or not sec.blocks:
            continue
        breadcrumb = " > ".join(f"{n} {t}".strip() for n, t in sec.path)
        header = f"{meta['spec']} (Rel-{meta['release']}) | {breadcrumb}\n\n"
        budget = MAX_TOKENS * CHARS_PER_TOKEN - len(header)
        pieces = split_blocks(sec.blocks, budget, OVERLAP_TOKENS * CHARS_PER_TOKEN)
        parent = sec.path[-2][0] if len(sec.path) > 1 else None
        key = f"{path.stem.split('-')[0]}-{sec.number or sec.title}".replace(" ", "_")
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key = f"{key}~{seen[key]}"
        for i, piece in enumerate(pieces):
            text = header + piece
            chunks.append({
                "chunk_id": f"{key}-{i}",
                **meta,
                "section": sec.number,
                "title": sec.title,
                "parent_section": parent,
                "section_path": breadcrumb,
                "level": sec.level,
                "part": i,
                "n_parts": len(pieces),
                "n_tokens": approx_tokens(text),
                "text": text,
            })
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="docx files (default: corpus/raw/*.docx)")
    ap.add_argument("-o", "--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    files = args.files or sorted(RAW_DIR.glob("*.docx"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as f:
        for path in files:
            if path.name.startswith(".~lock"):
                continue
            chunks = chunk_document(path)
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            sections = len({c["section"] for c in chunks})
            avg = sum(c["n_tokens"] for c in chunks) // max(len(chunks), 1)
            print(f"{path.name}: {sections} sections -> {len(chunks)} chunks (avg ~{avg} tokens)")
            total += len(chunks)
    print(f"Wrote {total} chunks to {args.out}")


if __name__ == "__main__":
    main()
