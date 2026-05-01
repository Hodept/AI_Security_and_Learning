from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingestion_script import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_QDRANT_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    ask_ollama,
    build_prompt,
    get_qdrant_client,
    run_query_pipeline,
)


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
DEBUG_RAG = os.getenv("DEBUG_RAG", "").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG_RAG else "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rag_api")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_optional(name: str) -> str | None:
    value = os.getenv(name)
    return value or None


class RagRequest(BaseModel):
    query: str = Field(min_length=1)
    collection: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
    qdrant_api_key: str | None = Field(default_factory=lambda: env_optional("QDRANT_API_KEY"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    top_k: int = Field(default_factory=lambda: env_int("TOP_K", DEFAULT_TOP_K), ge=1, le=25)
    candidate_k: int = Field(default_factory=lambda: env_int("CANDIDATE_K", DEFAULT_CANDIDATE_K), ge=1, le=100)
    filter_json: str | None = None
    reranker_model: str | None = None
    timeout: float = Field(default_factory=lambda: env_float("QDRANT_TIMEOUT", DEFAULT_TIMEOUT), gt=0)


class AskRequest(RagRequest):
    ollama_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    ollama_timeout: float = Field(default_factory=lambda: env_float("OLLAMA_TIMEOUT", DEFAULT_OLLAMA_TIMEOUT), gt=0)


class ChunkResponse(BaseModel):
    id: str
    text: str
    score: float
    citation: str
    payload: dict[str, Any]
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    mode: Literal["search"] = "search"
    query: str
    chunks: list[ChunkResponse]


class AskResponse(BaseModel):
    mode: Literal["ask"] = "ask"
    query: str
    answer: str
    chunks: list[ChunkResponse]


class HealthResponse(BaseModel):
    status: str
    qdrant_url: str
    collection: str
    points: int | None = None
    debug_enabled: bool = False


class ErrorDetail(BaseModel):
    message: str
    error_type: str
    stage: str
    debug_enabled: bool
    traceback: str | None = None


app = FastAPI(title="Local PDF RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    logger.info("request method=%s path=%s", request.method, request.url.path)
    response = await call_next(request)
    logger.info(
        "response method=%s path=%s status_code=%s",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


def error_detail(error: Exception, stage: str) -> dict[str, Any]:
    detail = ErrorDetail(
        message=str(error),
        error_type=type(error).__name__,
        stage=stage,
        debug_enabled=DEBUG_RAG,
        traceback=traceback.format_exc() if DEBUG_RAG else None,
    )
    return detail.dict()


def chunk_to_response(chunk: Any) -> ChunkResponse:
    return ChunkResponse(
        id=chunk.id,
        text=chunk.text,
        score=chunk.score,
        citation=chunk.citation,
        payload=chunk.payload,
        metadata=chunk.metadata,
    )


def retrieve_chunks(request: RagRequest) -> list[ChunkResponse]:
    logger.info(
        "retrieval started collection=%s top_k=%s candidate_k=%s model=%s",
        request.collection,
        request.top_k,
        request.candidate_k,
        request.embedding_model,
    )
    try:
        chunks = run_query_pipeline(
            query=request.query,
            qdrant_url=request.qdrant_url,
            api_key=request.qdrant_api_key,
            collection_name=request.collection,
            embedding_model_name=request.embedding_model,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            filter_json=request.filter_json,
            reranker_model=request.reranker_model,
            timeout=request.timeout,
        )
    except Exception as error:
        logger.exception("retrieval failed collection=%s", request.collection)
        raise HTTPException(status_code=500, detail=error_detail(error, "retrieval")) from error

    logger.info("retrieval completed collection=%s chunks=%s", request.collection, len(chunks))
    return [chunk_to_response(chunk) for chunk in chunks]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_url = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    api_key = env_optional("QDRANT_API_KEY")
    timeout = env_float("QDRANT_TIMEOUT", DEFAULT_TIMEOUT)

    try:
        client = get_qdrant_client(qdrant_url, api_key, timeout)
        client.get_collections()
        count = None
        if client.collection_exists(collection_name=collection):
            count = client.count(collection_name=collection, exact=True).count
    except Exception as error:
        logger.exception("health check failed qdrant_url=%s collection=%s", qdrant_url, collection)
        raise HTTPException(status_code=503, detail=error_detail(error, "health")) from error

    return HealthResponse(
        status="ok",
        qdrant_url=qdrant_url,
        collection=collection,
        points=count,
        debug_enabled=DEBUG_RAG,
    )


@app.post("/search", response_model=SearchResponse)
def search(request: RagRequest) -> SearchResponse:
    logger.info("search requested query_length=%s", len(request.query))
    chunks = retrieve_chunks(request)
    return SearchResponse(query=request.query, chunks=chunks)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    logger.info(
        "ask requested query_length=%s ollama_model=%s",
        len(request.query),
        request.ollama_model,
    )
    chunks = retrieve_chunks(request)

    try:
        prompt = build_prompt(request.query, chunks)
        logger.info("ollama generation started model=%s chunks=%s", request.ollama_model, len(chunks))
        answer = ask_ollama(
            prompt=prompt,
            model=request.ollama_model,
            ollama_url=request.ollama_url,
            timeout=request.ollama_timeout,
        )
    except Exception as error:
        logger.exception("ollama generation failed model=%s", request.ollama_model)
        raise HTTPException(status_code=500, detail=error_detail(error, "ollama_generation")) from error

    logger.info("ollama generation completed model=%s answer_length=%s", request.ollama_model, len(answer))
    return AskResponse(query=request.query, answer=answer, chunks=chunks)
