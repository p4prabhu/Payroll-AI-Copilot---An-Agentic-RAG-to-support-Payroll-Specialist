"""Create reviewable chunks from extracted payroll documents."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "knowledge/extracted_v2"
OUTPUT_DIR = ROOT / "knowledge/chunks"
TARGET_CHARACTERS = 2800
OVERLAP_CHARACTERS = 400
TABLE_ROWS_PER_CHUNK = 10


def split_guidance(text: str) -> list[str]:
    """Group paragraphs and sentences without cutting their text."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    units = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= TARGET_CHARACTERS:
            units.append(paragraph)
        else:
            units.extend(
                re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
            )

    chunks = []
    current = []

    for unit in units:
        candidate = "\n\n".join(current + [unit])
        if current and len(candidate) > TARGET_CHARACTERS:
            chunks.append("\n\n".join(current))
            overlap = current[-1]
            current = (
                [overlap]
                if len(overlap) <= OVERLAP_CHARACTERS
                else []
            )
        current.append(unit)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def make_chunk(
    document: dict[str, Any],
    page: dict[str, Any],
    kind: str,
    index: int,
    content: str,
    **details: Any,
) -> dict[str, Any]:
    """Attach source provenance and an identifier to a chunk."""
    identity = (
        f"{document['document_id']}:{document['sha256']}:"
        f"{page['pdf_page_number']}:{kind}:{index}:{content}"
    )
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    warnings = list(page["warnings"])

    if len(content) > TARGET_CHARACTERS:
        warnings.append(
            "Oversized chunk retained intact; review before embedding."
        )

    return {
        "chunk_id": chunk_id,
        "kind": kind,
        "content": content,
        "pdf_page_number": page["pdf_page_number"],
        "source_metadata": document,
        "warnings": warnings,
        "review_status": "pending",
        **details,
    }


def chunk_page(
    document: dict[str, Any],
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preserve deduction-table rows and protect formula pages."""
    text = page["text"].strip()
    chunks = []

    if document["document_type"] == "deduction_table":
        if page["warnings"]:
            raise ValueError(
                "Deduction page has extraction warnings; review it first."
            )

        if len(page["tables"]) != 1:
            raise ValueError("Expected one deduction table per page")

        table = page["tables"][0]
        if len(table) < 3:
            raise ValueError("Table lacks headers or data rows")

        headers = table[:2]
        rows = table[2:]
        page_intro = "\n".join(text.splitlines()[:5])
        footnote = re.search(r"(?m)^\(\d+\)\s", text)
        notes = text[footnote.start():] if footnote else ""

        for offset in range(0, len(rows), TABLE_ROWS_PER_CHUNK):
            batch = rows[offset:offset + TABLE_ROWS_PER_CHUNK]
            content = (
                f"{document['title']}\n{page_intro}\n"
                "Headers and rows are ordered arrays. "
                "Blank cells remain blank; do not infer values.\n"
                + json.dumps(
                    {"headers": headers, "rows": batch},
                    ensure_ascii=False,
                )
            )
            if notes:
                content += "\nPage footnotes:\n" + notes

            chunks.append(
                make_chunk(
                    document,
                    page,
                    "deduction_table",
                    offset // TABLE_ROWS_PER_CHUNK,
                    content,
                    headers=headers,
                    rows=batch,
                    data_row_start=offset + 1,
                    data_row_end=offset + len(batch),
                )
            )
        return chunks

    formula_page = bool(
        re.search(r"(?m)^\s*[A-Z][A-Z0-9*]*\s*=", text)
    )
    sections = [text] if formula_page else split_guidance(text)

    for index, section in enumerate(sections):
        chunks.append(
            make_chunk(
                document,
                page,
                "formula_page" if formula_page else "guidance",
                index,
                section,
                adjacent_page_context_may_be_needed=True,
                contains_unreviewed_tables=bool(page["tables"]),
            )
        )

    return chunks


def main() -> None:
    """Write one candidate chunk file per extracted document."""
    inputs = sorted(INPUT_DIR.glob("*.json"))
    if not inputs:
        raise FileNotFoundError("No extracted documents found")

    results = []

    for path in inputs:
        extracted = json.loads(path.read_text(encoding="utf-8"))
        if extracted["schema_version"] != 2:
            raise ValueError(f"Unsupported extraction schema: {path.name}")

        chunks = []
        for page in extracted["pages"]:
            chunks.extend(
                chunk_page(extracted["source_metadata"], page)
            )

        destination = OUTPUT_DIR / path.name
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite {destination.name}"
            )
        results.append((destination, chunks))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for destination, chunks in results:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(
                {"schema_version": 1, "chunks": chunks},
                stream,
                indent=2,
                ensure_ascii=False,
            )
            stream.write("\n")

        print(f"{destination.name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
