"""Prepare PDF chunks and embeddings for local Qdrant ingestion.

Examples:
    python3 tokenize_pdfs.py ./paper.pdf ./handbook.pdf --output ./qdrant_points.jsonl
    python3 tokenize_pdfs.py ./pdfs/*.pdf --chunk-tokens 512 --overlap-tokens 96

The output is JSON Lines in a Qdrant-friendly point format:
    {"id": "...", "vector": [...], "payload": {"text": "...", "metadata": {...}}}

PDF is the only implemented input format for now. File handling is intentionally
isolated so additional loaders can be added without changing chunking/storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol


DEFAULT_MODEL = "sentence-transformers/multi-qa-distilbert-cos-v1"
DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 96
DEFAULT_BATCH_SIZE = 32


@dataclass
class DocumentBlock:
    """A semantically meaningful text block extracted from a source document."""

    text: str
    file_path: Path
    file_name: str
    file_type: str
    page: int
    block_index: int
    section_title: str | None
    block_type: str


@dataclass
class PreparedPoint:
    """One Qdrant-ready vector point."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


class TokenizingModel(Protocol):
    tokenizer: Any


def missing_dependency_message(package: str) -> str:
    return (
        f"Missing dependency: {package}. Install dependencies with:\n"
        "python3 -m pip install pypdf sentence-transformers qdrant-client\n"
        "Run that command after activating the same virtualenv you use for this script."
    )


def normalize_text(text: str) -> str:
    """Clean common PDF extraction artifacts while keeping paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def detect_block_type(text: str) -> str:
    compact = text.strip()
    if is_heading(compact):
        return "heading"
    if re.match(r"^(\d+[\.)]|[-*•])\s+", compact):
        return "list_item"
    return "paragraph"


def is_heading(text: str) -> bool:
    if len(text) > 120 or len(text.split()) > 14:
        return False
    if re.match(r"^(\d+(\.\d+)*\.?\s+|[A-Z][A-Z0-9 /\-:]{4,}$)", text):
        return True
    return text.istitle() and not text.endswith((".", "?", "!"))


def load_pdf_blocks(pdf_path: Path) -> list[DocumentBlock]:
    """Extract PDF text into page-aware semantic blocks."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError(missing_dependency_message("pypdf")) from error

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    blocks: list[DocumentBlock] = []
    current_section: str | None = None
    block_index = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if not page_text:
            continue

        for paragraph in split_paragraphs(page_text):
            block_type = detect_block_type(paragraph)
            if block_type == "heading":
                current_section = paragraph

            blocks.append(
                DocumentBlock(
                    text=paragraph,
                    file_path=pdf_path,
                    file_name=pdf_path.name,
                    file_type="pdf",
                    page=page_number,
                    block_index=block_index,
                    section_title=current_section,
                    block_type=block_type,
                )
            )
            block_index += 1

    if not blocks:
        raise ValueError(f"No extractable text found in: {pdf_path}")

    return blocks


def load_document_blocks(file_path: Path) -> list[DocumentBlock]:
    """Dispatch input files to format-specific loaders.

    Add future loaders here, for example DOCX, HTML, Markdown, or plain text.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_blocks(file_path)
    raise ValueError(f"Unsupported file type for now: {file_path}")


def validate_input_files(file_paths: Iterable[Path]) -> list[Path]:
    paths = list(file_paths)
    if not paths:
        raise ValueError("No input files were provided.")

    unsupported = [path for path in paths if path.suffix.lower() != ".pdf"]
    if unsupported:
        examples = ", ".join(str(path) for path in unsupported[:5])
        raise ValueError(
            "tokenize_pdfs.py only supports PDF files right now. "
            f"Unsupported input file(s): {examples}"
        )

    missing = [path for path in paths if not path.exists()]
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Input file(s) not found: {examples}")

    return paths


def count_text_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def count_tokens(model: TokenizingModel, texts: list[str]) -> list[int]:
    tokenizer = model.tokenizer
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def split_oversized_block(tokenizer: Any, block: DocumentBlock, max_tokens: int) -> list[DocumentBlock]:
    """Split one very large block on sentence boundaries before chunk assembly."""
    if count_text_tokens(tokenizer, block.text) <= max_tokens:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block.text)
    split_blocks: list[DocumentBlock] = []
    buffer: list[str] = []

    for sentence in sentences:
        candidate = " ".join([*buffer, sentence]).strip()
        if buffer and count_text_tokens(tokenizer, candidate) > max_tokens:
            split_blocks.append(
                DocumentBlock(
                    text=" ".join(buffer),
                    file_path=block.file_path,
                    file_name=block.file_name,
                    file_type=block.file_type,
                    page=block.page,
                    block_index=block.block_index,
                    section_title=block.section_title,
                    block_type=block.block_type,
                )
            )
            buffer = [sentence]
        else:
            buffer.append(sentence)

    if buffer:
        split_blocks.append(
            DocumentBlock(
                text=" ".join(buffer),
                file_path=block.file_path,
                file_name=block.file_name,
                file_type=block.file_type,
                page=block.page,
                block_index=block.block_index,
                section_title=block.section_title,
                block_type=block.block_type,
            )
        )

    return split_blocks


def build_chunk_text(blocks: list[DocumentBlock]) -> str:
    parts: list[str] = []
    active_section: str | None = None
    for block in blocks:
        if block.section_title and block.section_title != active_section:
            active_section = block.section_title
            if block.block_type != "heading":
                parts.append(active_section)
        parts.append(block.text)
    return "\n\n".join(parts).strip()


def chunk_blocks(
    blocks: list[DocumentBlock],
    model: TokenizingModel,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """Create token-aware sliding-window chunks that preserve block boundaries."""
    if overlap_tokens >= chunk_tokens:
        raise ValueError("--overlap-tokens must be smaller than --chunk-tokens")

    tokenizer = model.tokenizer
    normalized_blocks: list[DocumentBlock] = []
    for block in blocks:
        normalized_blocks.extend(split_oversized_block(tokenizer, block, chunk_tokens))

    chunks: list[dict[str, Any]] = []
    window: list[DocumentBlock] = []
    window_tokens = 0
    chunk_index_by_file: dict[Path, int] = {}

    for block in normalized_blocks:
        block_tokens = count_text_tokens(tokenizer, block.text)
        if window and window_tokens + block_tokens > chunk_tokens:
            chunks.append(make_chunk(window, model, chunk_index_by_file))
            window, window_tokens = trim_window_for_overlap(window, tokenizer, overlap_tokens)

        window.append(block)
        window_tokens += block_tokens

    if window:
        chunks.append(make_chunk(window, model, chunk_index_by_file))

    return chunks


def trim_window_for_overlap(
    window: list[DocumentBlock],
    tokenizer: Any,
    overlap_tokens: int,
) -> tuple[list[DocumentBlock], int]:
    if overlap_tokens <= 0:
        return [], 0

    kept: list[DocumentBlock] = []
    kept_tokens = 0
    for block in reversed(window):
        block_tokens = count_text_tokens(tokenizer, block.text)
        if kept and kept_tokens + block_tokens > overlap_tokens:
            break
        kept.insert(0, block)
        kept_tokens += block_tokens
    return kept, kept_tokens


def make_chunk(
    blocks: list[DocumentBlock],
    model: TokenizingModel,
    chunk_index_by_file: dict[Path, int],
) -> dict[str, Any]:
    first_block = blocks[0]
    file_path = first_block.file_path
    chunk_index = chunk_index_by_file.get(file_path, 0)
    chunk_index_by_file[file_path] = chunk_index + 1

    text = build_chunk_text(blocks)
    token_count = count_text_tokens(model.tokenizer, text)
    pages = sorted({block.page for block in blocks})
    sections = list(dict.fromkeys(block.section_title for block in blocks if block.section_title))
    block_types = list(dict.fromkeys(block.block_type for block in blocks))
    stable_id = make_chunk_id(file_path, chunk_index, text)

    return {
        "id": stable_id,
        "text": text,
        "metadata": {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_type": first_block.file_type,
            "page_start": min(pages),
            "page_end": max(pages),
            "pages": pages,
            "chunk_index": chunk_index,
            "block_start": blocks[0].block_index,
            "block_end": blocks[-1].block_index,
            "section_titles": sections,
            "block_types": block_types,
            "token_count": token_count,
            "character_count": len(text),
            "chunking_strategy": "semantic_block_sliding_window",
        },
    }


def make_chunk_id(file_path: Path, chunk_index: int, text: str) -> str:
    source = f"{file_path.resolve()}:{chunk_index}:{text}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(missing_dependency_message("sentence-transformers")) from error

    return SentenceTransformer(model_name)


def prepare_files_for_qdrant(
    file_paths: Iterable[Path],
    model_name: str = DEFAULT_MODEL,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    normalize_embeddings: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[PreparedPoint]:
    """Extract, chunk, tokenize, and embed files into Qdrant point records."""
    paths = validate_input_files(file_paths)
    model = load_embedding_model(model_name)
    chunks: list[dict[str, Any]] = []

    for file_path in paths:
        blocks = load_document_blocks(file_path)
        chunks.extend(chunk_blocks(blocks, model, chunk_tokens, overlap_tokens))

    if not chunks:
        raise ValueError("No chunks were created from the provided files.")

    texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )

    prepared_points: list[PreparedPoint] = []
    created_at = datetime.now(UTC).isoformat()
    for chunk, embedding in zip(chunks, embeddings):
        metadata = dict(chunk["metadata"])
        metadata["embedding_model"] = model_name
        metadata["embedding_normalized"] = normalize_embeddings
        metadata["created_at"] = created_at

        prepared_points.append(
            PreparedPoint(
                id=str(chunk["id"]),
                vector=embedding.tolist(),
                payload={
                    "text": str(chunk["text"]),
                    "metadata": metadata,
                    "source": metadata["source"],
                    "file_name": metadata["file_name"],
                    "file_type": metadata["file_type"],
                    "page_start": metadata["page_start"],
                    "page_end": metadata["page_end"],
                    "chunk_index": metadata["chunk_index"],
                    "section_titles": metadata["section_titles"],
                    "token_count": metadata["token_count"],
                },
            )
        )

    return prepared_points


def write_jsonl(points: Iterable[PreparedPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for point in points:
            output_file.write(json.dumps(asdict(point), ensure_ascii=False) + "\n")


def upsert_points_to_qdrant(
    points: list[PreparedPoint],
    collection_name: str,
    url: str,
    distance: str,
) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError as error:
        raise ImportError(missing_dependency_message("qdrant-client")) from error

    if not points:
        raise ValueError("No points to upload.")

    distance_value = getattr(Distance, distance.upper())
    client = QdrantClient(url=url)
    vector_size = len(points[0].vector)
    existing = [collection.name for collection in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_value),
        )

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=point.id, vector=point.vector, payload=point.payload)
            for point in points
        ],
    )


def default_output_path(files: list[Path]) -> Path:
    if len(files) == 1:
        return files[0].with_name(f"{files[0].stem}_qdrant_points.jsonl")
    return Path("qdrant_points.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PDFs, create semantic sliding-window chunks, and prepare Qdrant points."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more PDF files to process.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="JSONL output path. Defaults to <pdf_stem>_qdrant_points.jsonl or ./qdrant_points.jsonl.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"Target maximum tokens per chunk. Default: {DEFAULT_CHUNK_TOKENS}",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help=f"Sliding-window overlap tokens. Default: {DEFAULT_OVERLAP_TOKENS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable normalized embeddings. Keep enabled for cosine-similarity Qdrant collections.",
    )
    parser.add_argument(
        "--upload-qdrant",
        action="store_true",
        help="Also upsert generated points into a local Qdrant collection.",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant URL for --upload-qdrant. Default: http://localhost:6333",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name for --upload-qdrant. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--distance",
        choices=("cosine", "dot", "euclid", "manhattan"),
        default="cosine",
        help="Qdrant vector distance for collection creation. Default: cosine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.files)

    prepared_points = prepare_files_for_qdrant(
        file_paths=args.files,
        model_name=args.model,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        normalize_embeddings=not args.no_normalize_embeddings,
        batch_size=args.batch_size,
    )
    write_jsonl(prepared_points, output_path)

    if args.upload_qdrant:
        upsert_points_to_qdrant(
            points=prepared_points,
            collection_name=args.collection,
            url=args.qdrant_url,
            distance=args.distance,
        )

    token_counts = [int(point.payload["metadata"]["token_count"]) for point in prepared_points]
    processed_files = sorted({str(point.payload["source"]) for point in prepared_points})
    print(f"Processed files: {len(processed_files)}")
    for file_path in processed_files:
        print(f"- {file_path}")
    print(f"Qdrant points written: {len(prepared_points)}")
    print(f"Token range: {min(token_counts)}-{max(token_counts)}")
    print(f"Output: {output_path}")
    if args.upload_qdrant:
        print(f"Uploaded to Qdrant collection '{args.collection}' at {args.qdrant_url}")


if __name__ == "__main__":
    main()
