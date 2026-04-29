"""Import JSONL embeddings into Milvus and run semantic searches.

Milvus must be running before you use this script. For a local Docker setup,
start Milvus separately, then connect to it with the default URI below.

Examples:
    python3 milvus_search.py import ./paper_embeddings.jsonl
    python3 milvus_search.py search "What does the document say about access control?"
    python3 milvus_search.py search "incident response" --limit 5 --show-vector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterable


DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_MODEL = "sentence-transformers/multi-qa-distilbert-cos-v1"
DEFAULT_URI = "http://localhost:19530"
DEFAULT_BATCH_SIZE = 100
DEFAULT_TIMEOUT = 10.0
VECTOR_FIELD = "embedding"


def missing_dependency_message(package: str) -> str:
    return (
        f"Missing dependency: {package}. Install dependencies with:\n"
        "python3 -m pip install pymilvus sentence-transformers"
    )


def get_milvus_client(uri: str, token: str | None, timeout: float):
    try:
        from pymilvus import MilvusClient
    except ImportError as error:
        raise ImportError(missing_dependency_message("pymilvus")) from error

    if token:
        return MilvusClient(uri=uri, token=token, timeout=timeout)
    return MilvusClient(uri=uri, timeout=timeout)


def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(missing_dependency_message("sentence-transformers")) from error

    return SentenceTransformer(model_name)


def read_jsonl(jsonl_path: Path) -> Iterable[dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            validate_record(record, line_number)
            yield record


def validate_record(record: dict[str, Any], line_number: int) -> None:
    required_fields = ("id", "text", "embedding", "metadata")
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ValueError(f"Line {line_number} is missing required field(s): {fields}")

    if not isinstance(record["embedding"], list) or not record["embedding"]:
        raise ValueError(f"Line {line_number} has an empty or invalid embedding")

    if not isinstance(record["metadata"], dict):
        raise ValueError(f"Line {line_number} metadata must be a JSON object")


def collection_exists(client: Any, collection_name: str) -> bool:
    if hasattr(client, "has_collection"):
        return bool(client.has_collection(collection_name=collection_name))
    return collection_name in client.list_collections()


def create_collection_if_needed(
    client: Any,
    collection_name: str,
    dimension: int,
    metric_type: str,
    drop_existing: bool,
) -> None:
    if collection_exists(client, collection_name):
        if not drop_existing:
            return
        client.drop_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        vector_field_name=VECTOR_FIELD,
        metric_type=metric_type,
        auto_id=True,
        enable_dynamic_field=True,
    )


def convert_record_for_milvus(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        VECTOR_FIELD: record["embedding"],
        "chunk_id": record["id"],
        "text": record["text"],
        "source": metadata.get("source"),
        "file_name": metadata.get("file_name"),
        "page": metadata.get("page"),
        "chunk_index": metadata.get("chunk_index"),
        "token_count": metadata.get("token_count"),
        "character_count": metadata.get("character_count"),
        "embedding_model": metadata.get("embedding_model"),
        "metadata_json": json.dumps(metadata),
    }


def import_jsonl_to_milvus(
    jsonl_path: Path,
    uri: str,
    token: str | None,
    timeout: float,
    collection_name: str,
    metric_type: str,
    batch_size: int,
    drop_existing: bool,
) -> None:
    client = get_milvus_client(uri, token, timeout)
    records = read_jsonl(jsonl_path)

    first_record = next(records, None)
    if first_record is None:
        raise ValueError(f"No records found in: {jsonl_path}")

    dimension = len(first_record["embedding"])
    create_collection_if_needed(
        client=client,
        collection_name=collection_name,
        dimension=dimension,
        metric_type=metric_type,
        drop_existing=drop_existing,
    )

    total_inserted = 0
    batch = [convert_record_for_milvus(first_record)]
    expected_dimension = dimension

    for record in records:
        if len(record["embedding"]) != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for chunk {record['id']}: "
                f"expected {expected_dimension}, got {len(record['embedding'])}"
            )

        batch.append(convert_record_for_milvus(record))
        if len(batch) >= batch_size:
            total_inserted += insert_batch(client, collection_name, batch)
            batch = []

    if batch:
        total_inserted += insert_batch(client, collection_name, batch)

    flush_collection(client, collection_name)
    print(f"Imported {total_inserted} chunks into Milvus collection '{collection_name}'")
    print(f"URI: {uri}")
    print(f"Embedding dimension: {dimension}")


def insert_batch(client: Any, collection_name: str, batch: list[dict[str, Any]]) -> int:
    result = client.insert(collection_name=collection_name, data=batch)
    if isinstance(result, dict) and "insert_count" in result:
        return int(result["insert_count"])
    return len(batch)


def flush_collection(client: Any, collection_name: str) -> None:
    if hasattr(client, "flush"):
        client.flush(collection_name=collection_name)


def search_milvus(
    query: str,
    uri: str,
    token: str | None,
    timeout: float,
    collection_name: str,
    model_name: str,
    metric_type: str,
    limit: int,
    filter_expression: str,
    show_vector: bool,
) -> None:
    client = get_milvus_client(uri, token, timeout)
    model = load_embedding_model(model_name)
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    output_fields = [
        "chunk_id",
        "text",
        "source",
        "file_name",
        "page",
        "chunk_index",
        "token_count",
        "embedding_model",
    ]
    if show_vector:
        output_fields.append(VECTOR_FIELD)

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field=VECTOR_FIELD,
        filter=filter_expression,
        limit=limit,
        output_fields=output_fields,
        search_params={"metric_type": metric_type, "params": {}},
    )

    if not results or not results[0]:
        print("No search results found.")
        return

    for index, hit in enumerate(results[0], start=1):
        entity = hit.get("entity", {})
        distance = hit.get("distance")
        preview = shorten(entity.get("text", "").replace("\n", " "), width=220, placeholder="...")

        print(f"\nResult {index}")
        print(f"score: {distance}")
        print(f"chunk_id: {entity.get('chunk_id')}")
        print(
            f"file: {entity.get('file_name')} "
            f"page: {entity.get('page')} "
            f"chunk: {entity.get('chunk_index')} "
            f"tokens: {entity.get('token_count')}"
        )
        print(f"text: {preview}")

        if show_vector:
            vector = entity.get(VECTOR_FIELD, [])
            print(f"vector_dimensions: {len(vector)}")
            print(f"vector_preview: {json.dumps(vector[:20])}")


def check_connection(uri: str, token: str | None, timeout: float) -> None:
    client = get_milvus_client(uri, token, timeout)
    collections = client.list_collections()
    print(f"Connected to Milvus at {uri}")
    print(f"Collections: {collections}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import PDF chunk embeddings into Milvus and search them."
    )
    parser.add_argument(
        "--uri",
        default=DEFAULT_URI,
        help=f"Milvus URI. Default: {DEFAULT_URI}",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Milvus auth token, if needed. Example for local default auth: root:Milvus",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Milvus collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--metric-type",
        default="COSINE",
        choices=["COSINE", "IP", "L2"],
        help="Vector similarity metric. Default: COSINE",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Milvus connection timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Import a JSONL embeddings file.")
    import_parser.add_argument("jsonl", type=Path, help="Path to JSONL file from text_spliter.py.")
    import_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Milvus insert batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    import_parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate the collection before import.",
    )

    search_parser = subparsers.add_parser("search", help="Search the Milvus collection.")
    search_parser.add_argument("query", help="Natural-language search query.")
    search_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model for query embeddings. Default: {DEFAULT_MODEL}",
    )
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results.")
    search_parser.add_argument(
        "--filter",
        default="",
        help='Optional Milvus scalar filter. Example: file_name == "paper.pdf"',
    )
    search_parser.add_argument(
        "--show-vector",
        action="store_true",
        help="Print the first 20 values of each returned vector.",
    )

    subparsers.add_parser("health", help="Check the Milvus connection.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "import":
        import_jsonl_to_milvus(
            jsonl_path=args.jsonl,
            uri=args.uri,
            token=args.token,
            timeout=args.timeout,
            collection_name=args.collection,
            metric_type=args.metric_type,
            batch_size=args.batch_size,
            drop_existing=args.drop_existing,
        )
    elif args.command == "search":
        search_milvus(
            query=args.query,
            uri=args.uri,
            token=args.token,
            timeout=args.timeout,
            collection_name=args.collection,
            model_name=args.model,
            metric_type=args.metric_type,
            limit=args.limit,
            filter_expression=args.filter,
            show_vector=args.show_vector,
        )
    elif args.command == "health":
        check_connection(args.uri, args.token, args.timeout)


if __name__ == "__main__":
    main()
