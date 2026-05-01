# Local PDF RAG Pipeline

This folder contains a small local Retrieval Augmented Generation pipeline for PDF files. The workflow is intentionally split into three stages so each script has one clear job:

1. `tokenize_pdfs.py` extracts PDF text, chunks it, embeds it, and writes Qdrant-ready JSONL.
2. `qdrant_upload.py` imports that JSONL into a local Qdrant collection and inspects what is loaded.
3. `ingestion_script.py` queries Qdrant, optionally reranks candidates, cites sources, and can ask a local Ollama model to answer from the retrieved context.

## Requirements

Create and activate a virtual environment, then install the Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pypdf sentence-transformers qdrant-client
```

Start Qdrant locally:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

For answer generation, install and start Ollama separately, then pull a model:

```bash
ollama pull llama3
```

## First-Time API and UI Setup With Docker Compose

The `UI_Creation/` folder contains a Docker Compose setup for running the local RAG API dependencies together. This is the recommended path when you want the same setup to work cleanly across machines.

The Compose stack starts:

- `ollama`: local LLM runtime, exposed on `http://localhost:11434`
- `qdrant`: vector database, exposed on `http://localhost:6333` and `localhost:6334`
- `rag-api`: FastAPI service for search and ask requests, exposed on `http://localhost:8000`
- `streamlit`: browser UI for the RAG query layer, exposed on `http://localhost:8501`
- `ollama-model-init`: one-time helper that pulls the configured Ollama model before the UI starts

Inside Docker, services communicate by container name:

```text
QDRANT_URL=http://qdrant:6333
OLLAMA_URL=http://ollama:11434
RAG_API_URL=http://rag-api:8000
```

From your host machine, the existing scripts still use localhost:

```text
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
RAG_API_URL=http://localhost:8000
```

Start the stack from the `UI_Creation/` folder:

```bash
cd UI_Creation
docker compose up --build
```

You can also start the stack from any directory by passing the Compose file path:

```bash
docker compose \
  --project-name ai-security-rag \
  -f /path/to/AI_Security_and_Learning/UI_Creation/docker-compose.yml \
  up --build
```

The Compose file uses paths relative to `UI_Creation/docker-compose.yml`, so `context: ..` and `..:/app` resolve to the surrounding `AI_Security_and_Learning/` folder no matter where the command is launched from. Use a stable `--project-name` to avoid accidentally reusing containers from another copy of the project.

The first run may take a while because Docker needs to download images, Python packages, the SentenceTransformer embedding model, and the default Ollama model. By default, the stack uses the same settings as the scripts:

```text
QDRANT_COLLECTION=pdf_chunks
EMBEDDING_MODEL=sentence-transformers/multi-qa-distilbert-cos-v1
OLLAMA_MODEL=llama3
TOP_K=5
CANDIDATE_K=20
QDRANT_TIMEOUT=30
OLLAMA_TIMEOUT=300
RAG_API_REQUEST_TIMEOUT=330
DEBUG_RAG=false
LOG_LEVEL=INFO
```

To use a different Ollama model, set `OLLAMA_MODEL` before starting Compose:

```bash
OLLAMA_MODEL=mistral docker compose up --build
```

After the stack is running, prepare and upload your PDF data from the main `AI_Security_and_Learning/` folder:

```bash
cd ..
python3 tokenize_pdfs.py ./PDFs/*.pdf --output qdrant_points.jsonl
python3 qdrant_upload.py --url http://localhost:6333 import qdrant_points.jsonl --verify
python3 qdrant_upload.py --url http://localhost:6333 sources
```

Then open the Streamlit UI:

```text
http://localhost:8501
```

Streamlit talks to the FastAPI service, and the FastAPI service talks to Qdrant and Ollama:

```text
Streamlit -> rag-api -> Qdrant + Ollama
```

The Streamlit sidebar includes query controls for slower hosts or larger collections:

- `Mode`: use `Search` to test retrieval without waiting for Ollama generation.
- `Top K`: number of chunks returned to the answer prompt. Lower values are faster.
- `Candidate K`: initial retrieval pool before optional reranking. Lower values reduce retrieval work.
- `Qdrant timeout seconds`: how long the API waits for Qdrant retrieval.
- `Ollama timeout seconds`: how long the API waits for local model generation.
- `UI request timeout seconds`: how long Streamlit waits for the FastAPI response.
- `Metadata filter JSON`: optional Qdrant equality filter, for example `{"file_name":"policy.pdf"}`.
- `Reranker model`: optional CrossEncoder reranker. This can improve relevance but adds CPU work.

For underpowered hosts, start with:

```text
Mode=Search
Top K=2
Candidate K=5
Ollama timeout seconds=900
UI request timeout seconds=930
```

Then switch to `Ask Ollama` after retrieval is working.

You can also test the same running services from the command line:

```bash
python3 ingestion_script.py \
  --qdrant-url http://localhost:6333 \
  ask "Summarize incident response requirements" \
  --ollama-url http://localhost:11434 \
  --ollama-model llama3
```

Or call the FastAPI service directly:

```bash
curl http://localhost:8000/health
```

Search without answer generation:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the material say about access control?","top_k":5,"candidate_k":20}'
```

Retrieve context and ask Ollama:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize incident response requirements","ollama_model":"llama3","top_k":3,"candidate_k":10}'
```

Stop the stack when you are done:

```bash
cd UI_Creation
docker compose down
```

Qdrant data, Ollama models, and model caches are stored in Docker volumes, so they remain available across restarts.

## `rag_api.py`

Intent: expose the RAG query layer as a FastAPI service. This keeps Streamlit focused on the UI and keeps the existing ingestion functions reusable from both the command line and HTTP.

What it does:

- Provides `GET /health` for Qdrant connectivity and collection point count when the collection exists.
- Provides `POST /search` to retrieve matching Qdrant chunks without asking Ollama.
- Provides `POST /ask` to retrieve matching chunks, build a grounded prompt, and ask Ollama for an answer.
- Uses the same defaults as `ingestion_script.py`: collection `pdf_chunks`, embedding model `sentence-transformers/multi-qa-distilbert-cos-v1`, `top_k=5`, and `candidate_k=20`.

Run the API locally without Docker:

```bash
python3 -m pip install fastapi "uvicorn[standard]" qdrant-client sentence-transformers
uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

The interactive API docs are available at:

```text
http://localhost:8000/docs
```

### API Debugging and Error Output

The FastAPI service logs each HTTP request, retrieval run, Qdrant failure, and Ollama generation failure. In normal mode, API errors include the failing stage, error type, and message:

```json
{
  "detail": {
    "message": "Collection pdf_chunks not found",
    "error_type": "UnexpectedResponse",
    "stage": "retrieval",
    "debug_enabled": false,
    "traceback": null
  }
}
```

Enable detailed debug output when you want tracebacks in API responses and more verbose service logs:

```bash
cd UI_Creation
DEBUG_RAG=true LOG_LEVEL=DEBUG docker compose up --build
```

View service logs:

```bash
docker compose logs -f rag-api
docker compose logs -f streamlit
docker compose logs -f qdrant
docker compose logs -f ollama
```

Streamlit also has a `Show debug payloads` checkbox in the sidebar. When enabled, the UI shows the request payload it sent to FastAPI and the full response payload it received. If an API call fails, Streamlit shows an `Error details` expander with the structured API error response.

Check API health:

```bash
curl http://localhost:8000/health
```

Inspect a failing API call with verbose curl output:

```bash
curl -v -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","collection":"pdf_chunks"}'
```

Example with longer timeouts and fewer chunks:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize incident response requirements","top_k":2,"candidate_k":5,"timeout":60,"ollama_timeout":900}'
```

## End-To-End Flow

Run these from the directory that contains the scripts and your `PDFs/` folder:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf --output qdrant_points.jsonl
python3 qdrant_upload.py import qdrant_points.jsonl --verify
python3 qdrant_upload.py sources
python3 ingestion_script.py search "What does the material say about access control?"
python3 ingestion_script.py ask "Summarize incident response requirements" --ollama-model llama3
```

## `tokenize_pdfs.py`

Intent: convert one or more PDFs into embedded Qdrant point records. This script does not upload to Qdrant. It creates a JSONL file that `qdrant_upload.py` imports later.

What it does:

- Validates that inputs are PDF files.
- Extracts page text with `pypdf`.
- Preserves document metadata such as source path, file name, page range, chunk index, section titles, token count, and character count.
- Uses semantic block chunking with a sliding token window.
- Generates embeddings with a SentenceTransformer model.
- Writes JSONL records shaped like Qdrant points: `id`, `vector`, and `payload`.

Basic usage:

```bash
python3 tokenize_pdfs.py ./PDFs/example.pdf
```

Multiple PDFs:

```bash
python3 tokenize_pdfs.py ./PDFs/policy.pdf ./PDFs/handbook.pdf
```

Glob a folder of PDFs:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf
```

### `tokenize_pdfs.py` Options

Show help:

```bash
python3 tokenize_pdfs.py --help
```

Set an output JSONL file:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf --output qdrant_points.jsonl
```

Short form for output:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf -o qdrant_points.jsonl
```

Use a specific SentenceTransformer embedding model:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf --model sentence-transformers/multi-qa-distilbert-cos-v1
```

Set target chunk size in tokens:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf --chunk-tokens 512
```

Set sliding-window overlap in tokens:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf --overlap-tokens 96
```

Set embedding batch size:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf --batch-size 16
```

Disable normalized embeddings:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf --no-normalize-embeddings
```

Full example using every non-help option:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf \
  --output qdrant_points.jsonl \
  --model sentence-transformers/multi-qa-distilbert-cos-v1 \
  --chunk-tokens 512 \
  --overlap-tokens 96 \
  --batch-size 32
```

## `qdrant_upload.py`

Intent: import tokenizer output into Qdrant and inspect loaded data. This script does not perform user-facing RAG queries. Use `ingestion_script.py` for search and Ollama answers.

What it does:

- Reads JSONL generated by `tokenize_pdfs.py`.
- Validates point IDs, vectors, payloads, and metadata.
- Creates a Qdrant collection when needed.
- Checks vector dimensions before importing.
- Performs idempotent upserts using stable point IDs.
- Skips duplicate IDs within the input JSONL.
- Verifies import results.
- Summarizes loaded source files.

Global options can be placed before the command:

```bash
python3 qdrant_upload.py --url http://localhost:6333 --collection pdf_chunks health
```

### `qdrant_upload.py` Global Options

Show help:

```bash
python3 qdrant_upload.py --help
```

Use a custom Qdrant URL:

```bash
python3 qdrant_upload.py --url http://localhost:6333 health
```

Use a Qdrant API key:

```bash
python3 qdrant_upload.py --api-key local-dev-key health
```

Use a custom collection:

```bash
python3 qdrant_upload.py --collection security_docs health
```

Set Qdrant timeout:

```bash
python3 qdrant_upload.py --timeout 20 health
```

### Import Command

Import a JSONL file:

```bash
python3 qdrant_upload.py import qdrant_points.jsonl
```

Show import help:

```bash
python3 qdrant_upload.py import --help
```

Set upsert batch size:

```bash
python3 qdrant_upload.py import qdrant_points.jsonl --batch-size 50
```

Choose Qdrant distance metric:

```bash
python3 qdrant_upload.py import qdrant_points.jsonl --distance cosine
python3 qdrant_upload.py import qdrant_points.jsonl --distance dot
python3 qdrant_upload.py import qdrant_points.jsonl --distance euclid
python3 qdrant_upload.py import qdrant_points.jsonl --distance manhattan
```

Delete and recreate the collection before import:

```bash
python3 qdrant_upload.py import qdrant_points.jsonl --recreate
```

Verify after import:

```bash
python3 qdrant_upload.py import qdrant_points.jsonl --verify
```

Full import example using every import option:

```bash
python3 qdrant_upload.py \
  --url http://localhost:6333 \
  --collection pdf_chunks \
  --timeout 20 \
  import qdrant_points.jsonl \
  --batch-size 100 \
  --distance cosine \
  --verify
```

Use `--recreate` only when you intentionally want to delete the existing collection:

```bash
python3 qdrant_upload.py --collection pdf_chunks import qdrant_points.jsonl --recreate --verify
```

### Verify Command

Confirm that data is present:

```bash
python3 qdrant_upload.py verify
```

Show verify help:

```bash
python3 qdrant_upload.py verify --help
```

Scroll a sample of stored points:

```bash
python3 qdrant_upload.py verify --sample 5
```

### Sources Command

Show how many distinct sources are loaded:

```bash
python3 qdrant_upload.py sources
```

Show sources help:

```bash
python3 qdrant_upload.py sources --help
```

Set scroll batch size:

```bash
python3 qdrant_upload.py sources --batch-size 200
```

Scan only the first N points:

```bash
python3 qdrant_upload.py sources --limit 1000
```

Full sources example:

```bash
python3 qdrant_upload.py --collection pdf_chunks sources --batch-size 200 --limit 1000
```

### Health Command

Check Qdrant connectivity:

```bash
python3 qdrant_upload.py health
```

Health with global options:

```bash
python3 qdrant_upload.py --url http://localhost:6333 --timeout 20 health
```

## `ingestion_script.py`

Intent: interact with the loaded Qdrant collection. This is the RAG query layer.

What it does:

- Embeds user queries with the configured SentenceTransformer model.
- Retrieves top candidates from Qdrant.
- Applies optional metadata filters.
- Returns top-k chunks.
- Optionally reranks candidates with a CrossEncoder model.
- Prints source attribution and citations.
- Builds a grounded context prompt.
- Sends the prompt to a local Ollama model for answer generation.
- Leaves a `QueryStage` interface for future agent routing, memory, planning, or tool layers.

Global options can be placed before the command:

```bash
python3 ingestion_script.py --collection pdf_chunks search "access control"
```

### `ingestion_script.py` Global Options

Show help:

```bash
python3 ingestion_script.py --help
```

Use a custom Qdrant URL:

```bash
python3 ingestion_script.py --qdrant-url http://localhost:6333 health
```

Use a Qdrant API key:

```bash
python3 ingestion_script.py --api-key local-dev-key health
```

Use a custom Qdrant collection:

```bash
python3 ingestion_script.py --collection security_docs health
```

Use a specific query embedding model:

```bash
python3 ingestion_script.py --embedding-model sentence-transformers/multi-qa-distilbert-cos-v1 search "access control"
```

Set Qdrant timeout:

```bash
python3 ingestion_script.py --timeout 45 search "access control"
```

### Search Command

Retrieve candidates without asking Ollama:

```bash
python3 ingestion_script.py search "What does the material say about access control?"
```

Show search help:

```bash
python3 ingestion_script.py search --help
```

Set final result count:

```bash
python3 ingestion_script.py search "access control" --top-k 3
```

Set initial candidate pool before reranking:

```bash
python3 ingestion_script.py search "access control" --candidate-k 20
```

Apply a metadata filter:

```bash
python3 ingestion_script.py search "access control" --filter '{"file_name":"policy.pdf"}'
```

Use a reranker model:

```bash
python3 ingestion_script.py search "access control" --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2
```

Print full chunk text:

```bash
python3 ingestion_script.py search "access control" --show-text
```

Full search example using every search option:

```bash
python3 ingestion_script.py \
  --qdrant-url http://localhost:6333 \
  --collection pdf_chunks \
  --embedding-model sentence-transformers/multi-qa-distilbert-cos-v1 \
  --timeout 45 \
  search "access control" \
  --top-k 5 \
  --candidate-k 20 \
  --filter '{"file_name":"policy.pdf"}' \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --show-text
```

### Ask Command

Retrieve context and ask a local Ollama model:

```bash
python3 ingestion_script.py ask "Summarize incident response requirements" --ollama-model llama3
```

Show ask help:

```bash
python3 ingestion_script.py ask --help
```

Set final result count:

```bash
python3 ingestion_script.py ask "Summarize incident response requirements" --top-k 3
```

Set initial candidate pool before reranking:

```bash
python3 ingestion_script.py ask "Summarize incident response requirements" --candidate-k 10
```

Apply a metadata filter:

```bash
python3 ingestion_script.py ask "Summarize access control" --filter '{"file_name":"policy.pdf"}'
```

Use a reranker model:

```bash
python3 ingestion_script.py ask "Summarize access control" --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2
```

Use a custom Ollama URL:

```bash
python3 ingestion_script.py ask "Summarize access control" --ollama-url http://localhost:11434
```

Use a specific Ollama model:

```bash
python3 ingestion_script.py ask "Summarize access control" --ollama-model llama3
```

Increase Ollama generation timeout:

```bash
python3 ingestion_script.py ask "Summarize access control" --ollama-timeout 300
```

Show retrieved context before the answer:

```bash
python3 ingestion_script.py ask "Summarize access control" --show-context
```

Full ask example using every ask option:

```bash
python3 ingestion_script.py \
  --qdrant-url http://localhost:6333 \
  --collection pdf_chunks \
  --embedding-model sentence-transformers/multi-qa-distilbert-cos-v1 \
  --timeout 45 \
  ask "Summarize access control requirements" \
  --top-k 3 \
  --candidate-k 10 \
  --filter '{"file_name":"policy.pdf"}' \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --ollama-url http://localhost:11434 \
  --ollama-model llama3 \
  --ollama-timeout 300 \
  --show-context
```

### Health Command

Check Qdrant collection status:

```bash
python3 ingestion_script.py health
```

Health with global options:

```bash
python3 ingestion_script.py --qdrant-url http://localhost:6333 --collection pdf_chunks --timeout 45 health
```

## Common Troubleshooting

Missing `sentence_transformers`:

```text
ModuleNotFoundError: No module named 'sentence_transformers'
```

Install dependencies inside the active virtual environment:

```bash
source .venv/bin/activate
python3 -m pip install pypdf sentence-transformers qdrant-client
```

Accidentally passing non-PDF files:

```bash
python3 tokenize_pdfs.py ./PDFs/*.py
```

Use:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf
```

Ollama timeout:

```text
Error: Ollama timed out after ...
```

Try fewer chunks or a longer timeout:

```bash
python3 ingestion_script.py ask "your question" --top-k 2 --candidate-k 5 --ollama-timeout 300
```

Hugging Face unauthenticated warning:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This is not usually the failure. It means the embedding model is being downloaded or checked without an HF token. Once cached, the model should load locally. Add an HF token only if you need higher rate limits or private models.

Qdrant not running:

```bash
python3 qdrant_upload.py health
```

If it fails, start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Recommended Defaults

For a local RAG setup using cosine similarity:

```bash
python3 tokenize_pdfs.py ./PDFs/*.pdf \
  --output qdrant_points.jsonl \
  --chunk-tokens 512 \
  --overlap-tokens 96

python3 qdrant_upload.py import qdrant_points.jsonl --distance cosine --verify

python3 ingestion_script.py ask "your question" \
  --ollama-model llama3 \
  --top-k 3 \
  --candidate-k 10 \
  --ollama-timeout 300
```
