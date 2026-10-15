"""Extract catalogued payroll PDFs into page-based JSON files."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "knowledge/sources/catalog.json"
OUTPUT_DIR = PROJECT_ROOT / "knowledge/extracted_v2"


def extract_tables(
    page: Any,
    document: dict[str, Any],
) -> tuple[list[Any], str, list[str]]:
    """Use a tailored strategy for CRA deduction-table layouts."""
    if document["document_type"] != "deduction_table":
        return page.extract_tables(), "default", []

    finder = page.debug_tablefinder()
    vertical = [
        edge for edge in finder.edges
        if edge["orientation"] == "v"
        and edge["height"] > page.height / 2
    ]
    horizontal = [
        edge for edge in finder.edges
        if edge["orientation"] == "h"
        and edge["width"] > page.width * 0.8
    ]

    if not vertical or not horizontal:
        return (
            page.extract_tables(),
            "default_fallback",
            ["Expected deduction-table boundaries not found."],
        )

    left = min(edge["x0"] for edge in horizontal)
    right = max(edge["x1"] for edge in horizontal)
    top = min(edge["top"] for edge in vertical)
    bottom = max(edge["bottom"] for edge in vertical)
    boundaries = sorted(
        {left, right, *(edge["x0"] for edge in vertical)}
    )

    cropped = page.crop((left, top, right, bottom))
    tables = cropped.extract_tables(
        {
            "vertical_strategy": "explicit",
            "explicit_vertical_lines": boundaries,
            "horizontal_strategy": "text",
            "intersection_x_tolerance": 40,
        }
    )
    tables = [
        [
            row for row in table
            if any(cell and cell.strip() for cell in row)
        ]
        for table in tables
    ]
    tables = [table for table in tables if table]

    income_tax = any(
        topic in document["topics"]
        for topic in ("federal_income_tax", "Ontario_income_tax")
    )
    expected_columns = 12 if income_tax else 8
    warnings = []

    if len(tables) != 1 or any(
        len(row) != expected_columns
        for table in tables for row in table
    ):
        warnings.append("Unexpected deduction-table shape.")

    if income_tax and not any(
        "CC 10" in (cell or "")
        for table in tables for row in table for cell in row
    ):
        warnings.append("Claim-code header CC 10 is missing.")

    if any(
        sum(bool(re.search(r"\d", line)) for line in cell.splitlines()) > 1
        for table in tables for row in table for cell in row
        if cell
    ):
        warnings.append("Possible merged numeric rows.")

    return tables, "explicit_columns_text_rows", warnings


def extract_document(
    document: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    """Verify a PDF and extract its pages for later review."""
    pdf_path = (source_root / document["file_path"]).resolve()

    if not pdf_path.is_relative_to(source_root):
        raise ValueError("PDF path is outside the source directory")

    checksum = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if checksum != document["sha256"]:
        raise ValueError(f"Checksum mismatch: {pdf_path.name}")

    reader = PdfReader(pdf_path)
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        if len(reader.pages) != len(pdf.pages):
            raise ValueError("PDF readers disagree on page count")

        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            tables, strategy, warnings = extract_tables(
                pdf.pages[number - 1], document
            )

            if not text.strip():
                warnings.append(
                    "No text extracted; inspect for scans or blank pages."
                )

            pages.append(
                {
                    "pdf_page_number": number,
                    "text": text,
                    "tables": tables,
                    "table_extraction_strategy": strategy,
                    "warnings": warnings,
                    "review_status": "pending",
                }
            )
            pdf.pages[number - 1].close()

    return {
        "schema_version": 2,
        "source_metadata": document,
        "extraction_tools": {
            "pypdf": __import__("pypdf").__version__,
            "pdfplumber": pdfplumber.__version__,
        },
        "review_status": "pending",
        "page_count": len(pages),
        "pages": pages,
    }


def main() -> int:
    """Extract every catalogued PDF without replacing existing output."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    source_root = CATALOG_PATH.parent.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for document in catalog["documents"]:
        filename = Path(document["file_path"]).stem + ".json"
        output_path = OUTPUT_DIR / filename

        try:
            if output_path.exists():
                raise FileExistsError(
                    f"{filename} already exists; refusing to overwrite"
                )

            result = extract_document(document, source_root)

            with output_path.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, ensure_ascii=False)
                stream.write("\n")

            print(
                f"Extracted {document['document_id']}: "
                f"{result['page_count']} pages",
                flush=True,
            )
        except Exception as error:
            failures += 1
            print(
                f"Failed {document['document_id']}: {error}",
                flush=True,
            )

    print(f"Finished with {failures} failed documents.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
