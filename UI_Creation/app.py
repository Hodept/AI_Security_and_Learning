from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


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


RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-api:8000")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/multi-qa-distilbert-cos-v1",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K = env_int("TOP_K", 5)
CANDIDATE_K = env_int("CANDIDATE_K", 20)
QDRANT_TIMEOUT = env_float("QDRANT_TIMEOUT", 30.0)
OLLAMA_TIMEOUT = env_float("OLLAMA_TIMEOUT", 300.0)
RAG_API_REQUEST_TIMEOUT = env_float("RAG_API_REQUEST_TIMEOUT", 330.0)
DEBUG_RAG = os.getenv("DEBUG_RAG", "").lower() in {"1", "true", "yes", "on"}


def parse_error_response(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {"message": response.text}

    detail = body.get("detail", body)
    if isinstance(detail, dict):
        return detail
    return {"message": str(detail)}


def show_api_error(response: requests.Response) -> None:
    detail = parse_error_response(response)
    message = detail.get("message", response.reason)
    stage = detail.get("stage", "api_request")
    error_type = detail.get("error_type", f"HTTP {response.status_code}")

    st.error(f"{stage} failed: {error_type}: {message}")
    with st.expander("Error details"):
        st.json(detail)


st.set_page_config(page_title="Local RAG", page_icon=":material/search:", layout="wide")
st.title("Local PDF RAG")

with st.sidebar:
    st.subheader("Runtime")
    rag_api_url = st.text_input("RAG API URL", RAG_API_URL)
    collection = st.text_input("Collection", QDRANT_COLLECTION)
    embedding_model = st.text_input("Embedding model", EMBEDDING_MODEL)
    ollama_model = st.text_input("Ollama model", OLLAMA_MODEL)

    st.subheader("Retrieval")
    top_k = st.number_input("Top K", min_value=1, max_value=25, value=TOP_K)
    candidate_k = st.number_input("Candidate K", min_value=1, max_value=100, value=CANDIDATE_K)
    qdrant_timeout = st.number_input(
        "Qdrant timeout seconds",
        min_value=1.0,
        max_value=600.0,
        value=QDRANT_TIMEOUT,
        step=5.0,
    )
    filter_json = st.text_input("Metadata filter JSON", placeholder='{"file_name":"policy.pdf"}')
    reranker_model = st.text_input("Reranker model", placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2")

    st.subheader("Generation")
    ollama_timeout = st.number_input(
        "Ollama timeout seconds",
        min_value=10.0,
        max_value=3600.0,
        value=OLLAMA_TIMEOUT,
        step=30.0,
    )
    api_request_timeout = st.number_input(
        "UI request timeout seconds",
        min_value=10.0,
        max_value=3900.0,
        value=RAG_API_REQUEST_TIMEOUT,
        step=30.0,
    )
    show_debug = st.checkbox("Show debug payloads", value=DEBUG_RAG)

query = st.text_area("Question", height=120, placeholder="Ask a question about the loaded PDF chunks...")
mode = st.segmented_control("Mode", ["Search", "Ask Ollama"], default="Ask Ollama")

if st.button("Run", type="primary", disabled=not query.strip()):
    endpoint = "ask" if mode == "Ask Ollama" else "search"
    payload = {
        "query": query.strip(),
        "collection": collection,
        "embedding_model": embedding_model,
        "ollama_model": ollama_model,
        "top_k": int(top_k),
        "candidate_k": int(candidate_k),
        "timeout": float(qdrant_timeout),
        "ollama_timeout": float(ollama_timeout),
    }
    if filter_json.strip():
        payload["filter_json"] = filter_json.strip()
    if reranker_model.strip():
        payload["reranker_model"] = reranker_model.strip()

    try:
        with st.spinner("Calling the RAG API..."):
            response = requests.post(
                f"{rag_api_url.rstrip('/')}/{endpoint}",
                json=payload,
                timeout=float(api_request_timeout),
            )
            if not response.ok:
                show_api_error(response)
                st.stop()
            result = response.json()
    except requests.RequestException as error:
        st.error(f"RAG API request failed: {error}")
        with st.expander("Request details"):
            st.json({"url": f"{rag_api_url.rstrip('/')}/{endpoint}", "payload": payload})
        st.stop()

    if show_debug:
        with st.expander("Request payload"):
            st.json(payload)
        with st.expander("Response payload"):
            st.json(result)

    if mode == "Ask Ollama":
        st.subheader("Answer")
        st.write(result.get("answer", ""))

    st.subheader("Sources")
    for index, chunk in enumerate(result.get("chunks", []), start=1):
        citation = chunk.get("citation", "unknown source")
        score = float(chunk.get("score", 0.0))
        with st.expander(f"[{index}] {citation} | score={score:.4f}"):
            st.write(chunk.get("text", ""))
            st.caption(f"id={chunk.get('id')}")
