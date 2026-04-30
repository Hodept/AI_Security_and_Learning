"""Query local Qdrant RAG data and optionally answer with Ollama.

This script is the interaction layer that sits after:
    1. tokenize_pdfs.py creates embedded Qdrant points.
    2. qdrant_upload.py imports those points into local Qdrant.

Examples:
    python3 ingestion_script.py search "What does this say about access control?"
    python3 ingestion_script.py ask "Summarize incident response requirements" --ollama-model llama3
    python3 ingestion_script.py search "network segmentation" --filter '{"file_name":"policy.pdf"}'
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Protocol


DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/multi-qa-distilbert-cos-v1"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
DEFAULT_TIMEOUT = 30.0
DEFAULT_OLLAMA_TIMEOUT = 300.0


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    payload: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def citation(self) -> str:
        file_name = self.payload.get("file_name") or self.metadata.get("file_name") or "unknown"
        page_start = first_present(self.payload, self.metadata, "page_start")
        page_end = first_present(self.payload, self.metadata, "page_end")
        chunk_index = first_present(self.payload, self.metadata, "chunk_index")

        if page_start and page_end and page_start != page_end:
            page_text = f"pages {page_start}-{page_end}"
        elif page_start:
            page_text = f"page {page_start}"
        else:
            page_text = "page unknown"

        return f"{file_name}, {page_text}, chunk {chunk_index}"


class QueryStage(Protocol):
    """Future agent layers can implement this interface around retrieval."""

    def run(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...


class NoOpAgentStage:
    """Placeholder for future planning, routing, memory, or tool-use layers."""

    def run(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


def first_present(primary: dict[str, Any], secondary: dict[str, Any], key: str) -> Any:
    if key in primary and primary[key] is not None:
        return primary[key]
    return secondary.get(key)


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


def build_qdrant_filter(filter_json: str | None):
    if not filter_json:
        return None

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except ImportError as error:
        raise ImportError(missing_dependency_message("qdrant-client")) from error

    try:
        raw_filter = json.loads(filter_json)
    except json.JSONDecodeError as error:
        raise ValueError("--filter must be JSON, for example '{\"file_name\":\"paper.pdf\"}'") from error

    if not isinstance(raw_filter, dict):
        raise ValueError("--filter must be a JSON object")

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in raw_filter.items()
        ]
    )


def retrieve_candidates(
    query: str,
    qdrant_url: str,
    api_key: str | None,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
    candidate_k: int,
    filter_json: str | None,
    timeout: float,
) -> list[RetrievedChunk]:
    client = get_qdrant_client(qdrant_url, api_key, timeout)
    model = load_embedding_model(embedding_model_name)
    query_vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    qdrant_filter = build_qdrant_filter(filter_json)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=max(top_k, candidate_k),
        with_payload=True,
        with_vectors=False,
    ).points

    chunks: list[RetrievedChunk] = []
    for point in results:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        chunks.append(
            RetrievedChunk(
                id=str(point.id),
                text=str(payload.get("text", "")),
                score=float(point.score or 0.0),
                payload=payload,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )

    return chunks


def rerank_candidates(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    reranker_model: str | None,
) -> list[RetrievedChunk]:
    if not reranker_model:
        return chunks[:top_k]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise ImportError(missing_dependency_message("sentence-transformers")) from error

    reranker = CrossEncoder(reranker_model)
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = reranker.predict(pairs)

    reranked: list[RetrievedChunk] = []
    for chunk, score in zip(chunks, scores):
        reranked.append(
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                score=float(score),
                payload=chunk.payload,
                metadata=chunk.metadata,
            )
        )

    return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)[:top_k]


def build_context(chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[{index}] {chunk.citation}\n{chunk.text}")
    return "\n\n".join(context_parts)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "Answer the question using only the provided context. "
        "Cite sources inline with bracket numbers like [1]. "
        "If the context does not contain the answer, say that the indexed material does not say.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{build_context(chunks)}\n\n"
        "Answer:"
    )


def ask_ollama(prompt: str, model: str, ollama_url: str, timeout: float) -> str:
    request_body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError as error:
        raise TimeoutError(
            f"Ollama timed out after {timeout} seconds at {ollama_url}. "
            "Increase --ollama-timeout, use a smaller --top-k, or try a smaller local model."
        ) from error
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, TimeoutError):
            raise TimeoutError(
                f"Ollama timed out after {timeout} seconds at {ollama_url}. "
                "Increase --ollama-timeout, use a smaller --top-k, or try a smaller local model."
            ) from error
        raise ConnectionError(f"Could not reach Ollama at {ollama_url}: {error}") from error

    return str(payload.get("response", "")).strip()


def print_results(chunks: list[RetrievedChunk], show_text: bool) -> None:
    if not chunks:
        print("No candidates found.")
        return

    for index, chunk in enumerate(chunks, start=1):
        sections = chunk.metadata.get("section_titles") or []
        preview = shorten(chunk.text.replace("\n", " "), width=260, placeholder="...")

        print(f"\n[{index}] score={chunk.score:.4f}")
        print(f"source: {chunk.citation}")
        print(f"id: {chunk.id}")
        if sections:
            print(f"sections: {sections}")
        print(f"text: {chunk.text if show_text else preview}")


def print_citations(chunks: list[RetrievedChunk]) -> None:
    print("\nSources")
    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index}] {chunk.citation} | id={chunk.id}")


def run_query_pipeline(
    query: str,
    qdrant_url: str,
    api_key: str | None,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
    candidate_k: int,
    filter_json: str | None,
    reranker_model: str | None,
    timeout: float,
    agent_stage: QueryStage | None = None,
) -> list[RetrievedChunk]:
    candidates = retrieve_candidates(
        query=query,
        qdrant_url=qdrant_url,
        api_key=api_key,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        top_k=top_k,
        candidate_k=candidate_k,
        filter_json=filter_json,
        timeout=timeout,
    )
    reranked = rerank_candidates(query, candidates, top_k, reranker_model)
    stage = agent_stage or NoOpAgentStage()
    return stage.run(query, reranked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interact with Qdrant-backed RAG data and optionally answer with Ollama."
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help=f"Default: {DEFAULT_QDRANT_URL}")
    parser.add_argument("--api-key", default=None, help="Qdrant API key, if required.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help=f"Default: {DEFAULT_COLLECTION}")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help=f"Default: {DEFAULT_EMBEDDING_MODEL}")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"Default: {DEFAULT_TIMEOUT}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Retrieve top-k Qdrant candidates.")
    add_query_options(search_parser)
    search_parser.add_argument("--show-text", action="store_true", help="Print full chunk text instead of previews.")

    ask_parser = subparsers.add_parser("ask", help="Retrieve context and ask an Ollama model.")
    add_query_options(ask_parser)
    ask_parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"Default: {DEFAULT_OLLAMA_URL}")
    ask_parser.add_argument("--ollama-model", default="llama3", help="Local Ollama model name. Default: llama3")
    ask_parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=DEFAULT_OLLAMA_TIMEOUT,
        help=f"Ollama generation timeout in seconds. Default: {DEFAULT_OLLAMA_TIMEOUT}",
    )
    ask_parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before the answer.")

    subparsers.add_parser("health", help="Check Qdrant connectivity.")
    return parser.parse_args()


def add_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="User query.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Final candidates. Default: {DEFAULT_TOP_K}")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help=f"Initial candidates before reranking. Default: {DEFAULT_CANDIDATE_K}",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help='Qdrant equality filter as JSON. Example: \'{"file_name":"policy.pdf"}\'',
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Optional CrossEncoder reranker, for example cross-encoder/ms-marco-MiniLM-L-6-v2.",
    )


def health(qdrant_url: str, api_key: str | None, collection_name: str, timeout: float) -> None:
    client = get_qdrant_client(qdrant_url, api_key, timeout)
    info = client.get_collection(collection_name=collection_name)
    count = client.count(collection_name=collection_name, exact=True).count
    print(f"Connected to Qdrant at {qdrant_url}")
    print(f"collection: {collection_name}")
    print(f"status: {info.status}")
    print(f"points: {count}")


def main() -> None:
    args = parse_args()

    try:
        if args.command == "health":
            health(args.qdrant_url, args.api_key, args.collection, args.timeout)
            return

        chunks = run_query_pipeline(
            query=args.query,
            qdrant_url=args.qdrant_url,
            api_key=args.api_key,
            collection_name=args.collection,
            embedding_model_name=args.embedding_model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            filter_json=args.filter,
            reranker_model=args.reranker_model,
            timeout=args.timeout,
        )

        if args.command == "search":
            print_results(chunks, show_text=args.show_text)
            print_citations(chunks)
            return

        if args.command == "ask":
            if args.show_context:
                print_results(chunks, show_text=False)
            prompt = build_prompt(args.query, chunks)
            answer = ask_ollama(prompt, args.ollama_model, args.ollama_url, args.ollama_timeout)
            print("\nAnswer")
            print(answer)
            print_citations(chunks)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
