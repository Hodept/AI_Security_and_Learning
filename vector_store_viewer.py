"""Import JSONL embeddings into SQLite and inspect stored vector chunks.

Examples:
    python3 vector_store_viewer.py import ./paper_embeddings.jsonl --db ./vectors.db
    python3 vector_store_viewer.py list --db ./vectors.db
    python3 vector_store_viewer.py view <chunk_id> --db ./vectors.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from textwrap import shorten
from typing import Any


DEFAULT_DB_PATH = Path("embeddings.db")


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            metadata TEXT NOT NULL,
            source TEXT,
            file_name TEXT,
            page INTEGER,
            chunk_index INTEGER,
            token_count INTEGER,
            character_count INTEGER,
            embedding_model TEXT
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page)")
    connection.commit()


def import_jsonl(jsonl_path: Path, db_path: Path) -> None:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    with connect(db_path) as connection:
        create_schema(connection)
        imported_count = 0

        with jsonl_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                validate_record(record, line_number)
                metadata = record["metadata"]
                embedding = record["embedding"]

                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        id,
                        text,
                        embedding,
                        embedding_dimension,
                        metadata,
                        source,
                        file_name,
                        page,
                        chunk_index,
                        token_count,
                        character_count,
                        embedding_model
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["id"],
                        record["text"],
                        json.dumps(embedding),
                        len(embedding),
                        json.dumps(metadata),
                        metadata.get("source"),
                        metadata.get("file_name"),
                        metadata.get("page"),
                        metadata.get("chunk_index"),
                        metadata.get("token_count"),
                        metadata.get("character_count"),
                        metadata.get("embedding_model"),
                    ),
                )
                imported_count += 1

        connection.commit()

    print(f"Imported {imported_count} chunks into {db_path}")


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


def list_chunks(db_path: Path, limit: int) -> None:
    with connect(db_path) as connection:
        create_schema(connection)
        rows = connection.execute(
            """
            SELECT
                id,
                file_name,
                page,
                chunk_index,
                token_count,
                embedding_dimension,
                text
            FROM chunks
            ORDER BY file_name, page, chunk_index
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        print(f"No chunks found in {db_path}")
        return

    for row in rows:
        preview = shorten(row["text"].replace("\n", " "), width=90, placeholder="...")
        print(f"id: {row['id']}")
        print(
            "  "
            f"file={row['file_name']} page={row['page']} "
            f"chunk={row['chunk_index']} tokens={row['token_count']} "
            f"dimensions={row['embedding_dimension']}"
        )
        print(f"  text={preview}")


def view_chunk(db_path: Path, chunk_id: str, vector_limit: int | None) -> None:
    with connect(db_path) as connection:
        create_schema(connection)
        row = connection.execute(
            """
            SELECT id, text, embedding, metadata, embedding_dimension
            FROM chunks
            WHERE id = ?
            """,
            (chunk_id,),
        ).fetchone()

    if row is None:
        print(f"No chunk found with id: {chunk_id}")
        return

    embedding = json.loads(row["embedding"])
    metadata = json.loads(row["metadata"])
    displayed_embedding = embedding[:vector_limit] if vector_limit else embedding

    print(f"id: {row['id']}")
    print(f"embedding_dimension: {row['embedding_dimension']}")
    if vector_limit:
        print(f"embedding_preview_values: {len(displayed_embedding)}")
    print("metadata:")
    print(json.dumps(metadata, indent=2))
    print("text:")
    print(row["text"])
    print("embedding:")
    print(json.dumps(displayed_embedding, indent=2))


def stats(db_path: Path) -> None:
    with connect(db_path) as connection:
        create_schema(connection)
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS chunk_count,
                MIN(token_count) AS min_tokens,
                MAX(token_count) AS max_tokens,
                AVG(token_count) AS avg_tokens,
                MIN(embedding_dimension) AS min_dimensions,
                MAX(embedding_dimension) AS max_dimensions
            FROM chunks
            """
        ).fetchone()

    print(f"database: {db_path}")
    print(f"chunks: {row['chunk_count']}")
    print(f"token_range: {row['min_tokens']}-{row['max_tokens']}")
    print(f"average_tokens: {row['avg_tokens']:.2f}" if row["avg_tokens"] else "average_tokens: 0")
    print(f"embedding_dimensions: {row['min_dimensions']}-{row['max_dimensions']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import JSONL embedding chunks into SQLite and inspect them."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path. Default: {DEFAULT_DB_PATH}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Import a JSONL embeddings file.")
    import_parser.add_argument("jsonl", type=Path, help="Path to JSONL file from text_spliter.py.")

    list_parser = subparsers.add_parser("list", help="List stored chunks.")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of chunks to show.")

    view_parser = subparsers.add_parser("view", help="View one chunk and its vector.")
    view_parser.add_argument("chunk_id", help="Chunk id to inspect.")
    view_parser.add_argument(
        "--vector-limit",
        type=int,
        default=20,
        help="Number of vector values to print. Use 0 to print the full vector.",
    )

    subparsers.add_parser("stats", help="Show database summary statistics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "import":
        import_jsonl(args.jsonl, args.db)
    elif args.command == "list":
        list_chunks(args.db, args.limit)
    elif args.command == "view":
        vector_limit = args.vector_limit or None
        view_chunk(args.db, args.chunk_id, vector_limit)
    elif args.command == "stats":
        stats(args.db)


if __name__ == "__main__":
    main()
