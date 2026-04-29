"""Prepare PDF text chunks and embeddings for vector database storage.

Example:
    python text_spliter.py ./paper.pdf --output ./paper_embeddings.jsonl

The output is JSON Lines. Each line contains one chunk, its metadata, and its
embedding vector so it can be loaded into a vector database later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol


DEFAULT_MODEL = "sentence-transformers/multi-qa-distilbert-cos-v1"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


@dataclass
class PreparedChunk:
    """One vector-database-ready text chunk."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, object]


class TokenizingModel(Protocol):
    tokenizer: object


def missing_dependency_message(package: str) -> str:
    return (
        f"Missing dependency: {package}. Install dependencies with:\n"
        "python3 -m pip install pypdf langchain-text-splitters sentence-transformers"
    )


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, object]]:
    """Extract text from each page in a PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError(missing_dependency_message("pypdf")) from error

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, object]] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = normalize_text(text)
        if text:
            pages.append({"page": page_index, "text": text})

    if not pages:
        raise ValueError(f"No extractable text found in: {pdf_path}")

    return pages


def normalize_text(text: str) -> str:
    """Clean PDF extraction artifacts without changing the meaning."""
    lines = [line.strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line)


def build_text_splitter(chunk_size: int, chunk_overlap: int):
    """Create a splitter tuned for paragraph and sentence boundaries."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as error:
        raise ImportError(missing_dependency_message("langchain-text-splitters")) from error

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )


def split_pages(
    pages: Iterable[dict[str, object]],
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, object]]:
    """Split PDF pages into chunks while preserving page metadata."""
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    chunks: list[dict[str, object]] = []

    for page in pages:
        page_number = int(page["page"])
        page_text = str(page["text"])
        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            stable_id = make_chunk_id(pdf_path, page_number, chunk_index, chunk_text)
            chunks.append(
                {
                    "id": stable_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "character_count": len(chunk_text),
                    },
                }
            )

    return chunks


def make_chunk_id(pdf_path: Path, page_number: int, chunk_index: int, text: str) -> str:
    """Create a repeatable id for vector database upserts."""
    source = f"{pdf_path.resolve()}:{page_number}:{chunk_index}:{text}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def prepare_chunks_for_storage(
    pdf_path: Path,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    normalize_embeddings: bool = True,
) -> list[PreparedChunk]:
    """Extract, chunk, tokenize, and embed PDF content."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(missing_dependency_message("sentence-transformers")) from error

    pages = extract_pdf_pages(pdf_path)
    chunks = split_pages(pages, pdf_path, chunk_size, chunk_overlap)

    if not chunks:
        raise ValueError(f"No chunks were created from: {pdf_path}")

    model = SentenceTransformer(model_name)
    texts = [str(chunk["text"]) for chunk in chunks]
    token_counts = count_tokens(model, texts)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )

    prepared_chunks: list[PreparedChunk] = []
    for chunk, token_count, embedding in zip(chunks, token_counts, embeddings):
        metadata = dict(chunk["metadata"])
        metadata["token_count"] = token_count
        metadata["embedding_model"] = model_name

        prepared_chunks.append(
            PreparedChunk(
                id=str(chunk["id"]),
                text=str(chunk["text"]),
                embedding=embedding.tolist(),
                metadata=metadata,
            )
        )

    return prepared_chunks


def count_tokens(model: TokenizingModel, texts: list[str]) -> list[int]:
    """Count model-tokenized lengths for visibility before database insert."""
    tokenizer = model.tokenizer
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def write_jsonl(chunks: Iterable[PreparedChunk], output_path: Path) -> None:
    """Write chunks as JSON Lines for vector database import/upsert scripts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PDF text, split it, tokenize it, and embed it for vector storage."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file to process.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="JSONL output path. Defaults to <pdf_stem>_embeddings.jsonl.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum characters per chunk. Default: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlapping characters between chunks. Default: {DEFAULT_CHUNK_OVERLAP}",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable normalized embeddings. Keep enabled for cosine-similarity stores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.pdf.with_name(f"{args.pdf.stem}_embeddings.jsonl")

    prepared_chunks = prepare_chunks_for_storage(
        pdf_path=args.pdf,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        normalize_embeddings=not args.no_normalize_embeddings,
    )
    write_jsonl(prepared_chunks, output_path)

    token_counts = [int(chunk.metadata["token_count"]) for chunk in prepared_chunks]
    print(f"Processed: {args.pdf}")
    print(f"Chunks written: {len(prepared_chunks)}")
    print(f"Token range: {min(token_counts)}-{max(token_counts)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
