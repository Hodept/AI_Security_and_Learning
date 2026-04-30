"""Import tokenizer JSONL output into local Qdrant and run semantic searches.

Qdrant must be running before you use this script. For a local Docker setup:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Examples:
    python3 qdrant_upload.py import ./qdrant_points.jsonl --verify --sample-query "access control"
    python3 qdrant_upload.py search "What does the document say about incident response?"
    python3 qdrant_upload.py health
    python3 qdrant_upload.py sources
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterable


DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_MODEL = "sentence-transformers/multi-qa-distilbert-cos-v1"
DEFAULT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 100
DEFAULT_TIMEOUT = 10.0


@dataclass
class QdrantPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


def missing_dependency_message(package: str) -> str:
    return (
        f"Missing dependency: {package}. Install dependencies with:\n"
        "python3 -m pip install qdrant-client sentence-transformers"
    )


def get_qdrant_client(url: str, api_key: str | None, timeout: float):
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise ImportError(missing_dependency_message("qdrant-client")) from error

    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(missing_dependency_message("sentence-transformers")) from error

    return SentenceTransformer(model_name)


def read_jsonl(jsonl_path: Path) -> Iterable[QdrantPoint]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number} is not valid JSON: {error}") from error

            yield normalize_record(record, line_number)


def normalize_record(record: dict[str, Any], line_number: int) -> QdrantPoint:
    """Accept current Qdrant JSONL and the older embedding JSONL shape."""
    if {"id", "vector", "payload"}.issubset(record):
        vector = record["vector"]
        payload = record["payload"]
        point_id = record["id"]
    elif {"id", "embedding", "text", "metadata"}.issubset(record):
        vector = record["embedding"]
        metadata = record["metadata"]
        point_id = record["id"]
        payload = {
            "text": record["text"],
            "metadata": metadata,
            "source": metadata.get("source"),
            "file_name": metadata.get("file_name"),
            "file_type": metadata.get("file_type", "pdf"),
            "page_start": metadata.get("page_start", metadata.get("page")),
            "page_end": metadata.get("page_end", metadata.get("page")),
            "chunk_index": metadata.get("chunk_index"),
            "section_titles": metadata.get("section_titles", []),
            "token_count": metadata.get("token_count"),
        }
    else:
        raise ValueError(
            f"Line {line_number} must contain Qdrant fields "
            "(id, vector, payload) or legacy fields (id, embedding, text, metadata)"
        )

    if not isinstance(point_id, (str, int)):
        raise ValueError(f"Line {line_number} id must be a string or integer")
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"Line {line_number} has an empty or invalid vector")
    if not all(isinstance(value, int | float) for value in vector):
        raise ValueError(f"Line {line_number} vector must contain only numbers")
    if not isinstance(payload, dict):
        raise ValueError(f"Line {line_number} payload must be a JSON object")
    if "text" not in payload:
        raise ValueError(f"Line {line_number} payload is missing text")

    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
        payload["metadata"] = metadata
    if not isinstance(metadata, dict):
        raise ValueError(f"Line {line_number} payload.metadata must be a JSON object")

    return QdrantPoint(id=str(point_id), vector=[float(value) for value in vector], payload=payload)


def collection_exists(client: Any, collection_name: str) -> bool:
    return bool(client.collection_exists(collection_name=collection_name))


def ensure_collection(
    client: Any,
    collection_name: str,
    vector_size: int,
    distance: str,
    recreate: bool,
) -> None:
    try:
        from qdrant_client.models import Distance, VectorParams
    except ImportError as error:
        raise ImportError(missing_dependency_message("qdrant-client")) from error

    distance_value = getattr(Distance, distance.upper())
    if collection_exists(client, collection_name):
        info = client.get_collection(collection_name=collection_name)
        current_size = info.config.params.vectors.size
        if current_size != vector_size:
            if not recreate:
                raise ValueError(
                    f"Collection '{collection_name}' has vector size {current_size}, "
                    f"but input vectors have size {vector_size}. Use --recreate to rebuild it."
                )
            client.delete_collection(collection_name=collection_name)
        elif recreate:
            client.delete_collection(collection_name=collection_name)
        else:
            print(f"Collection '{collection_name}' already exists with vector size {current_size}")
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_value),
    )
    print(f"Created Qdrant collection '{collection_name}' size={vector_size} distance={distance}")


def upsert_points(
    client: Any,
    collection_name: str,
    points: list[QdrantPoint],
    batch_size: int,
) -> int:
    try:
        from qdrant_client.models import PointStruct
    except ImportError as error:
        raise ImportError(missing_dependency_message("qdrant-client")) from error

    total = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in batch
            ],
            wait=True,
        )
        total += len(batch)
        print(f"Upserted {total}/{len(points)} points")
    return total


def load_points(jsonl_path: Path) -> list[QdrantPoint]:
    points = list(read_jsonl(jsonl_path))
    if not points:
        raise ValueError(f"No points found in: {jsonl_path}")

    expected_size = len(points[0].vector)
    seen_ids: set[str] = set()
    duplicates = 0
    deduped: list[QdrantPoint] = []

    for point in points:
        if len(point.vector) != expected_size:
            raise ValueError(
                f"Vector dimension mismatch for point {point.id}: "
                f"expected {expected_size}, got {len(point.vector)}"
            )
        if point.id in seen_ids:
            duplicates += 1
            continue
        seen_ids.add(point.id)
        deduped.append(point)

    if duplicates:
        print(f"Skipped {duplicates} duplicate point id(s) from input")
    return deduped


def import_jsonl_to_qdrant(
    jsonl_path: Path,
    url: str,
    api_key: str | None,
    timeout: float,
    collection_name: str,
    distance: str,
    batch_size: int,
    recreate: bool,
    verify: bool,
    sample_query: str | None,
    model_name: str,
    limit: int,
) -> None:
    client = get_qdrant_client(url, api_key, timeout)
    points = load_points(jsonl_path)
    vector_size = len(points[0].vector)

    ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
        distance=distance,
        recreate=recreate,
    )
    upserted_count = upsert_points(client, collection_name, points, batch_size)

    print(f"Imported/upserted {upserted_count} point(s) into Qdrant collection '{collection_name}'")
    print(f"URL: {url}")
    print(f"Embedding dimension: {vector_size}")

    if verify:
        verify_collection(client, collection_name, expected_ids=[point.id for point in points[: min(10, len(points))]])

    if sample_query:
        print("\nSample query")
        search_qdrant(
            query=sample_query,
            url=url,
            api_key=api_key,
            timeout=timeout,
            collection_name=collection_name,
            model_name=model_name,
            limit=limit,
            query_filter=None,
            show_vector=False,
        )


def verify_collection(client: Any, collection_name: str, expected_ids: list[str] | None = None) -> None:
    info = client.get_collection(collection_name=collection_name)
    count = client.count(collection_name=collection_name, exact=True).count
    print(f"Collection status: {info.status}")
    print(f"Points present: {count}")

    if expected_ids:
        found = 0
        for point_id in expected_ids:
            records = client.retrieve(collection_name=collection_name, ids=[point_id], with_payload=False)
            if records:
                found += 1
        print(f"Verified imported ids: {found}/{len(expected_ids)}")

    records, _ = client.scroll(collection_name=collection_name, limit=1, with_payload=True, with_vectors=False)
    if records:
        payload = records[0].payload or {}
        metadata = payload.get("metadata", {})
        print("Sample stored point:")
        print(f"  id: {records[0].id}")
        print(f"  file: {payload.get('file_name')}")
        print(f"  pages: {payload.get('page_start')}-{payload.get('page_end')}")
        print(f"  chunk: {payload.get('chunk_index')} tokens: {payload.get('token_count')}")
        print(f"  metadata_keys: {', '.join(sorted(metadata.keys()))}")


def summarize_sources(
    client: Any,
    collection_name: str,
    batch_size: int,
    limit: int | None,
) -> None:
    """Scroll Qdrant payloads and summarize how many source files are loaded."""
    source_stats: dict[str, dict[str, Any]] = {}
    offset = None
    scanned_points = 0

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        for record in records:
            payload = record.payload or {}
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            source = (
                payload.get("source")
                or metadata.get("source")
                or payload.get("file_name")
                or metadata.get("file_name")
                or "unknown"
            )
            file_name = payload.get("file_name") or metadata.get("file_name") or Path(str(source)).name
            page_start = payload.get("page_start") or metadata.get("page_start")
            page_end = payload.get("page_end") or metadata.get("page_end")
            token_count = payload.get("token_count") or metadata.get("token_count") or 0

            stats = source_stats.setdefault(
                str(source),
                {
                    "file_name": file_name,
                    "chunks": 0,
                    "tokens": 0,
                    "page_min": None,
                    "page_max": None,
                },
            )
            stats["chunks"] += 1
            if isinstance(token_count, int | float):
                stats["tokens"] += int(token_count)
            if isinstance(page_start, int):
                stats["page_min"] = page_start if stats["page_min"] is None else min(stats["page_min"], page_start)
            if isinstance(page_end, int):
                stats["page_max"] = page_end if stats["page_max"] is None else max(stats["page_max"], page_end)

            scanned_points += 1
            if limit is not None and scanned_points >= limit:
                break

        if offset is None or (limit is not None and scanned_points >= limit):
            break

    collection_count = client.count(collection_name=collection_name, exact=True).count
    print(f"Collection: {collection_name}")
    print(f"Total points in collection: {collection_count}")
    print(f"Points scanned for source summary: {scanned_points}")
    print(f"Distinct sources loaded: {len(source_stats)}")

    if not source_stats:
        return

    print("\nSources:")
    for source, stats in sorted(source_stats.items(), key=lambda item: str(item[1]["file_name"])):
        page_min = stats["page_min"]
        page_max = stats["page_max"]
        if page_min is not None and page_max is not None:
            pages = f"{page_min}-{page_max}" if page_min != page_max else str(page_min)
        else:
            pages = "unknown"
        print(
            f"- {stats['file_name']} | chunks={stats['chunks']} "
            f"tokens={stats['tokens']} pages={pages}"
        )
        print(f"  source={source}")


def build_field_filter(filter_json: str | None):
    if not filter_json:
        return None

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except ImportError as error:
        raise ImportError(missing_dependency_message("qdrant-client")) from error

    try:
        raw_filter = json.loads(filter_json)
    except json.JSONDecodeError as error:
        raise ValueError("--filter must be a JSON object, for example '{\"file_name\":\"paper.pdf\"}'") from error

    if not isinstance(raw_filter, dict):
        raise ValueError("--filter must be a JSON object")

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in raw_filter.items()
        ]
    )


def search_qdrant(
    query: str,
    url: str,
    api_key: str | None,
    timeout: float,
    collection_name: str,
    model_name: str,
    limit: int,
    query_filter: str | None,
    show_vector: bool,
) -> None:
    client = get_qdrant_client(url, api_key, timeout)
    model = load_embedding_model(model_name)
    query_vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    qdrant_filter = build_field_filter(query_filter)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
        with_vectors=show_vector,
    ).points

    if not results:
        print("No search results found.")
        return

    for index, point in enumerate(results, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        preview = shorten(str(payload.get("text", "")).replace("\n", " "), width=220, placeholder="...")

        print(f"\nResult {index}")
        print(f"score: {point.score}")
        print(f"point_id: {point.id}")
        print(
            f"file: {payload.get('file_name')} "
            f"pages: {payload.get('page_start')}-{payload.get('page_end')} "
            f"chunk: {payload.get('chunk_index')} "
            f"tokens: {payload.get('token_count')}"
        )
        if metadata.get("section_titles"):
            print(f"sections: {metadata.get('section_titles')}")
        print(f"text: {preview}")

        if show_vector:
            vector = point.vector or []
            print(f"vector_dimensions: {len(vector)}")
            print(f"vector_preview: {json.dumps(vector[:20])}")


def check_connection(url: str, api_key: str | None, timeout: float) -> None:
    client = get_qdrant_client(url, api_key, timeout)
    collections = client.get_collections().collections
    print(f"Connected to Qdrant at {url}")
    if not collections:
        print("Collections: []")
        return
    print("Collections:")
    for collection in collections:
        print(f"- {collection.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import tokenizer JSONL points into Qdrant and search them."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Qdrant URL. Default: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Qdrant API key, if your local or remote instance requires one.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Qdrant connection timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Import tokenizer JSONL into Qdrant.")
    import_parser.add_argument("jsonl", type=Path, help="Path to JSONL file from tokenize_pdfs.py.")
    import_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Qdrant upsert batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    import_parser.add_argument(
        "--distance",
        choices=("cosine", "dot", "euclid", "manhattan"),
        default="cosine",
        help="Vector distance for collection creation. Default: cosine",
    )
    import_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before import.",
    )
    import_parser.add_argument(
        "--verify",
        action="store_true",
        help="Confirm the collection count and retrieve sample imported ids after import.",
    )
    import_parser.add_argument(
        "--sample-query",
        help="Run a semantic query immediately after import to display fresh results.",
    )
    import_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model for sample query embeddings. Default: {DEFAULT_MODEL}",
    )
    import_parser.add_argument("--limit", type=int, default=5, help="Number of sample query results.")

    search_parser = subparsers.add_parser("search", help="Search the Qdrant collection.")
    search_parser.add_argument("query", help="Natural-language search query.")
    search_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model for query embeddings. Default: {DEFAULT_MODEL}",
    )
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results.")
    search_parser.add_argument(
        "--filter",
        default=None,
        help='Optional Qdrant equality filter as JSON. Example: \'{"file_name":"paper.pdf"}\'',
    )
    search_parser.add_argument(
        "--show-vector",
        action="store_true",
        help="Print the first 20 values of each returned vector.",
    )

    verify_parser = subparsers.add_parser("verify", help="Confirm data is present in Qdrant.")
    verify_parser.add_argument("--sample", type=int, default=1, help="Number of sample points to scroll.")

    sources_parser = subparsers.add_parser("sources", help="Show how many distinct sources are loaded.")
    sources_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Qdrant scroll batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    sources_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of points to scan. Defaults to the full collection.",
    )

    subparsers.add_parser("health", help="Check the Qdrant connection.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.command == "import":
            import_jsonl_to_qdrant(
                jsonl_path=args.jsonl,
                url=args.url,
                api_key=args.api_key,
                timeout=args.timeout,
                collection_name=args.collection,
                distance=args.distance,
                batch_size=args.batch_size,
                recreate=args.recreate,
                verify=args.verify,
                sample_query=args.sample_query,
                model_name=args.model,
                limit=args.limit,
            )
        elif args.command == "search":
            search_qdrant(
                query=args.query,
                url=args.url,
                api_key=args.api_key,
                timeout=args.timeout,
                collection_name=args.collection,
                model_name=args.model,
                limit=args.limit,
                query_filter=args.filter,
                show_vector=args.show_vector,
            )
        elif args.command == "verify":
            client = get_qdrant_client(args.url, args.api_key, args.timeout)
            verify_collection(client, args.collection)
            records, _ = client.scroll(
                collection_name=args.collection,
                limit=args.sample,
                with_payload=True,
                with_vectors=False,
            )
            print(f"Scrolled sample points: {len(records)}")
        elif args.command == "sources":
            client = get_qdrant_client(args.url, args.api_key, args.timeout)
            summarize_sources(client, args.collection, args.batch_size, args.limit)
        elif args.command == "health":
            check_connection(args.url, args.api_key, args.timeout)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
