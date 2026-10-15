"""Index candidate payroll chunks using Perplexity and local Qdrant."""

import argparse
import base64
import json
import os
import struct
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "knowledge/chunks"
STORE_DIR = ROOT / "knowledge/vector_store/qdrant"
MODEL = "pplx-embed-v1-0.6b"
DIMENSIONS = 1024
COLLECTION_NAME = "ontario_payroll_pplx_v1_06b"

# Reviewed contents pages; PDF page numbers are one-based.
CONTENTS_PAGES = {
    "rc4120-25e": {4, 5},
    "t4001-25e": {4, 5},
    "t4127-01-26e": {3, 4},
    "t4127-jul-26e": {3},
    "t4130-25e": {4, 5},
}


def load_candidates() -> tuple[list[dict[str, Any]], int]:
    """Load chunks and exclude known contents pages."""
    chunks = []
    excluded = 0
    seen = set()

    for path in sorted(CHUNK_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        for chunk in data["chunks"]:
            document_id = chunk["source_metadata"]["document_id"]
            if chunk["pdf_page_number"] in CONTENTS_PAGES.get(
                document_id, set()
            ):
                excluded += 1
                continue

            if chunk["chunk_id"] in seen:
                raise ValueError("Duplicate chunk identifier")

            content = chunk["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty or invalid chunk text")

            # Conservative size guard, not an exact token count.
            # Oversized inputs stop here rather than being truncated.
            if len(content.encode("utf-8")) > 16000:
                raise ValueError("Chunk needs additional size review")

            seen.add(chunk["chunk_id"])
            chunks.append(chunk)

    if not chunks:
        raise ValueError("No candidate chunks found")

    return chunks, excluded


def embed(texts: list[str], key: str) -> list[list[float]]:
    """Request embeddings and validate response order and dimensions."""
    if sum(len(text.encode("utf-8")) for text in texts) > 60000:
        raise ValueError("Batch exceeds the conservative size guard")

    request = Request(
        "https://api.perplexity.ai/v1/embeddings",
        data=json.dumps(
            {
                "model": MODEL,
                "input": texts,
                "dimensions": DIMENSIONS,
                "encoding_format": "base64_int8",
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    if result["model"] != MODEL:
        raise ValueError("Unexpected embedding model")

    items = result["data"]
    if len(items) != len(texts):
        raise ValueError("Embedding count does not match input count")

    by_index = {item["index"]: item for item in items}
    if set(by_index) != set(range(len(texts))):
        raise ValueError("Invalid embedding response indexes")

    vectors = []
    for index in range(len(texts)):
        raw = base64.b64decode(
            by_index[index]["embedding"], validate=True
        )
        if len(raw) != DIMENSIONS:
            raise ValueError("Invalid embedding dimensions")

        vector = struct.unpack(f"{DIMENSIONS}b", raw)
        if not any(vector):
            raise ValueError("Embedding contains only zeros")
        vectors.append([float(value) for value in vector])

    return vectors


def metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Preserve citation and review metadata alongside each vector."""
    source = chunk["source_metadata"]
    result = {
        "document_id": source["document_id"],
        "title": source["title"],
        "source_url": source["source_url"],
        "source_page_url": source["source_page_url"],
        "publisher": source["publisher"],
        "source_sha256": source["sha256"],
        "pdf_page_number": chunk["pdf_page_number"],
        "kind": chunk["kind"],
        "topics": json.dumps(source["topics"]),
        "jurisdictions": json.dumps(source["jurisdictions"]),
        "review_status": chunk["review_status"],
        "needs_update_review": source["needs_update_review"],
        "warnings": json.dumps(chunk["warnings"]),
        "contains_unreviewed_tables": chunk.get(
            "contains_unreviewed_tables", False
        ),
        "adjacent_page_context_may_be_needed": chunk.get(
            "adjacent_page_context_may_be_needed", False
        ),
        "embedding_model": MODEL,
    }
    for field in (
        "edition_year",
        "effective_from",
        "effective_to",
        "pay_periods_per_year",
    ):
        if source.get(field) is not None:
            result[field] = source[field]

    return result


def point_id(chunk_id: str) -> str:
    """Map the full chunk hash to a stable Qdrant UUID."""
    return str(uuid5(NAMESPACE_URL, f"ontario-payroll:{chunk_id}"))


def payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        **metadata(chunk),
        "chunk_id": chunk["chunk_id"],
        "content": chunk["content"],
        "encoding_format": "base64_int8",
    }


def open_store() -> QdrantClient:
    """Open local storage and reject incompatible collection settings."""
    client = QdrantClient(path=str(STORE_DIR))
    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=DIMENSIONS, distance=models.Distance.COSINE
                ),
            )
        config = client.get_collection(COLLECTION_NAME).config.params.vectors
        if not isinstance(config, models.VectorParams) or (
            config.size != DIMENSIONS
            or config.distance != models.Distance.COSINE
        ):
            raise ValueError("Existing collection configuration differs")
        return client
    except Exception:
        client.close()
        raise


def pending_chunks(client: QdrantClient, chunks: list[dict]) -> list[dict]:
    """Validate existing payloads before skipping paid embedding requests."""
    identifiers = [point_id(chunk["chunk_id"]) for chunk in chunks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Qdrant point identifier collision")
    pending = []
    for start in range(0, len(chunks), 100):
        group = chunks[start:start + 100]
        existing = {
            str(record.id): record
            for record in client.retrieve(
                COLLECTION_NAME,
                ids=[point_id(chunk["chunk_id"]) for chunk in group],
                with_payload=True,
                with_vectors=False,
            )
        }
        for chunk in group:
            record = existing.get(point_id(chunk["chunk_id"]))
            if record is None:
                pending.append(chunk)
            elif record.payload != payload(chunk):
                raise ValueError("Existing record differs from its source chunk")
    return pending


def main() -> int:
    """Preview or index a limited batch of candidate chunks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    client = None
    try:
        chunks, excluded = load_candidates()
        print(f"Candidate chunks: {len(chunks)}")
        print(f"Contents-page chunks excluded: {excluded}")

        if args.dry_run:
            print("Dry run complete: no API calls or database writes.")
            return 0

        load_dotenv(ROOT / ".env", override=False)
        key = os.getenv("PERPLEXITY_API_KEY", "").strip()
        configured_model = os.getenv(
            "PERPLEXITY_EMBEDDING_MODEL", MODEL
        ).strip()
        if not key or configured_model != MODEL:
            raise ValueError("Missing key or unsupported model setting")

        client = open_store()
        pending = pending_chunks(client, chunks)

        selected = pending if args.all else pending[:args.limit]
        print(f"New chunks selected: {len(selected)}")

        # Eight inputs per request keeps initial batches small.
        for start in range(0, len(selected), 8):
            batch = selected[start:start + 8]
            vectors = embed(
                [chunk["content"] for chunk in batch], key
            )
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id(chunk["chunk_id"]),
                        vector=vector,
                        payload=payload(chunk),
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
                wait=True,
            )
            print(f"Stored batch: {len(batch)} chunks", flush=True)

        print(f"Collection records: {client.count(COLLECTION_NAME, exact=True).count}")
        return 0

    except HTTPError as error:
        print(f"Perplexity request failed: HTTP {error.code}.")
    except (URLError, TimeoutError):
        print("Network request failed or timed out.")
    except Exception:
        print("Indexing failed; check inputs and configuration.")
    finally:
        if client is not None:
            client.close()

    print("Completed stored batches are retained for the next run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
