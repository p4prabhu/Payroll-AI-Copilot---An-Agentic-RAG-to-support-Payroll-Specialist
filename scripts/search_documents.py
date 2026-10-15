"""Retrieve cited payroll evidence from Qdrant; no answers or calculations."""

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from index_documents import COLLECTION_NAME, DIMENSIONS, MODEL, ROOT, STORE_DIR, embed


DOCUMENT_TYPES = ("deduction_table", "fillable_form", "form", "guide")
PAY_PERIODS = (12, 24, 26, 52)


def build_filter(
    document_id: str | None,
    document_type: str | None,
    edition_year: int | None,
    pay_periods_per_year: int | None,
) -> models.Filter | None:
    """Combine explicit restrictions without treating edition year as validity."""
    if document_type is not None and document_type not in DOCUMENT_TYPES:
        raise ValueError("Unsupported document type")
    if edition_year is not None and (
        type(edition_year) is not int or not 1900 <= edition_year <= 2100
    ):
        raise ValueError("Edition year must be an integer between 1900 and 2100")
    if pay_periods_per_year is not None and (
        type(pay_periods_per_year) is not int or pay_periods_per_year not in PAY_PERIODS
    ):
        raise ValueError("Pay periods must be 12, 24, 26, or 52")

    conditions = []
    if document_id is not None:
        conditions.append(models.FieldCondition(
            key="document_id", match=models.MatchValue(value=document_id.strip())
        ))
    if document_type is not None:
        catalog = json.loads((ROOT / "knowledge/sources/catalog.json").read_text(encoding="utf-8"))
        identifiers = [
            document["document_id"] for document in catalog["documents"]
            if document["document_type"] == document_type
        ]
        conditions.append(models.FieldCondition(
            key="document_id", match=models.MatchAny(any=identifiers)
        ))
    for field, value in (
        ("edition_year", edition_year),
        ("pay_periods_per_year", pay_periods_per_year),
    ):
        if value is not None:
            conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    return models.Filter(must=conditions) if conditions else None


def search_documents(
    question: str, limit: int = 5, document_id: str | None = None,
    *, document_type: str | None = None, edition_year: int | None = None,
    pay_periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Return evidence for a question using the same model as the stored vectors.

    Scores indicate similarity, not answer confidence. Review status and dates
    remain attached to every result for the future agent's evidence checks.
    """
    question = question.strip()
    if not question or len(question.encode("utf-8")) > 16000:
        raise ValueError("Question must be nonempty and at most 16,000 UTF-8 bytes")
    if not 1 <= limit <= 20:
        raise ValueError("Result limit must be between 1 and 20")
    if document_id is not None and not document_id.strip():
        raise ValueError("Document identifier must be nonempty")
    query_filter = build_filter(document_id, document_type, edition_year, pay_periods_per_year)
    applied_filters = {
        field: value for field, value in {
            "document_id": document_id.strip() if document_id is not None else None,
            "document_type": document_type,
            "edition_year": edition_year,
            "pay_periods_per_year": pay_periods_per_year,
        }.items() if value is not None
    }
    if not (STORE_DIR / "meta.json").is_file():
        raise ValueError("Existing Qdrant database was not found; index documents first")

    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    configured_model = os.getenv("PERPLEXITY_EMBEDDING_MODEL", MODEL).strip()
    if not key or configured_model != MODEL:
        raise ValueError("Missing API key or unsupported embedding model setting")

    client = QdrantClient(path=str(STORE_DIR))
    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise ValueError("Payroll collection does not exist")
        config = client.get_collection(COLLECTION_NAME).config.params.vectors
        if not isinstance(config, models.VectorParams) or (
            config.size != DIMENSIONS or config.distance != models.Distance.COSINE
        ):
            raise ValueError("Payroll collection has incompatible vector settings")
        # Avoid paying for a query when the collection or filtered subset is empty.
        if client.count(COLLECTION_NAME, count_filter=query_filter, exact=True).count == 0:
            return {"question": question, "filters": applied_filters, "results": [], "status": "no_matches"}

        vector = embed([question], key)[0]
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
        results = []
        for rank, hit in enumerate(hits, start=1):
            evidence = dict(hit.payload or {})
            if evidence.get("embedding_model") != MODEL or (
                evidence.get("encoding_format") != "base64_int8"
            ):
                raise ValueError("Retrieved record uses incompatible embeddings")
            for field in ("chunk_id", "content", "title", "source_url", "pdf_page_number", "review_status"):
                if field not in evidence:
                    raise ValueError("Retrieved record is missing evidence metadata")
            for field in ("topics", "jurisdictions", "warnings"):
                evidence[field] = json.loads(evidence[field])
            results.append({"rank": rank, "similarity_score": float(hit.score), **evidence})
        return {
            "question": question,
            "filters": applied_filters,
            "status": "evidence_found" if results else "no_matches",
            "score_meaning": "Cosine similarity, not answer confidence",
            "results": results,
        }
    finally:
        client.close()


def main() -> int:
    # Government PDFs contain symbols outside Windows' default code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--document-id", help="Restrict search to a catalog document ID")
    parser.add_argument("--document-type", choices=DOCUMENT_TYPES)
    parser.add_argument("--edition-year", type=int, help="Publication edition year; does not establish effective-date applicability")
    parser.add_argument("--pay-periods", type=int, choices=PAY_PERIODS, help="Strict frequency filter; excludes documents without frequency metadata")
    parser.add_argument("--json", action="store_true", help="Print full text and metadata as JSON")
    args = parser.parse_args()
    try:
        result = search_documents(
            args.question, args.limit, args.document_id,
            document_type=args.document_type,
            edition_year=args.edition_year,
            pay_periods_per_year=args.pay_periods,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=True, indent=2))
        elif not result["results"]:
            print("No matching evidence found.")
            print(f"Applied filters: {result['filters']}")
        else:
            print("Retrieved evidence; similarity scores are not answer confidence.")
            print(f"Applied filters: {result['filters']}")
            for evidence in result["results"]:
                print(f"\n{evidence['rank']}. {evidence['title']}")
                print(f"PDF page {evidence['pdf_page_number']} | similarity {evidence['similarity_score']:.4f}")
                print(f"Source: {evidence['source_url']}")
                print(f"Review: {evidence['review_status']} | update review needed: {evidence['needs_update_review']}")
                print(f"Effective dates: {evidence.get('effective_from', 'unknown')} to {evidence.get('effective_to', 'unknown')}")
                print(f"Warnings: {evidence['warnings']}")
                print(evidence["content"][:800])
        return 0
    except HTTPError as error:
        print(f"Embedding request failed: HTTP {error.code}.", file=sys.stderr)
    except (URLError, TimeoutError):
        print("Embedding request failed or timed out.", file=sys.stderr)
    except ValueError as error:
        print(f"Search input or data check failed: {error}", file=sys.stderr)
    except Exception:
        print("Search failed; check local database access and configuration.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
