"""Check Perplexity embeddings without displaying credentials."""

import base64
import json
import os
import struct
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Send one sample text and validate the returned vector."""
    load_dotenv(ROOT / ".env", override=False)

    key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    model = os.getenv(
        "PERPLEXITY_EMBEDDING_MODEL", "pplx-embed-v1-0.6b"
    ).strip()

    if not key:
        print("Missing PERPLEXITY_API_KEY. Configure it locally.")
        return 1

    if model != "pplx-embed-v1-0.6b":
        print("This initial check supports pplx-embed-v1-0.6b.")
        return 1

    payload = {
        "model": model,
        "input": ["Ontario payroll deductions include CPP and EI."],
        "encoding_format": "base64_int8",
    }
    request = Request(
        "https://api.perplexity.ai/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)

        items = result["data"]
        if len(items) != 1 or items[0]["index"] != 0:
            raise ValueError("Unexpected embedding response")

        raw = base64.b64decode(
            items[0]["embedding"], validate=True
        )
        if len(raw) != 1024:
            raise ValueError("Unexpected vector dimensions")

        vector = struct.unpack(f"{len(raw)}b", raw)
        if not any(vector):
            raise ValueError("Returned vector contains only zeros")

        print("Embedding check: passed")
        print(f"Vector dimensions: {len(vector)}")

        tokens = result.get("usage", {}).get("total_tokens")
        if isinstance(tokens, int):
            print(f"Tokens used: {tokens}")

        return 0
    except HTTPError as error:
        print(f"API request failed: HTTP {error.code}.")
    except (URLError, TimeoutError):
        print("Network connection failed or timed out.")
    except (KeyError, TypeError, ValueError, struct.error):
        print("Embedding response validation failed.")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
