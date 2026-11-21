# RAG Chatbot for 3GPP 5G Specifications

Ask questions about 5G and get answers grounded **only** in two official 3GPP specifications, with a citation (spec + section) after every statement. When the documents don't cover a question, the chatbot says *"I could not find this information in the documents."*

- **Corpus:** TS 23.501 V18.12.0 (5G system architecture) and TS 24.501 V18.14.0 (NAS protocol), Release 18
- **Retrieval:** hybrid search (bge-base-en-v1.5 vectors in ChromaDB + BM25), Reciprocal Rank Fusion, cross-encoder reranker, routing by spec
- **Generation:** `openai/gpt-oss-120b` via Groq, with citation checks and explicit refusal
- **App:** React frontend + FastAPI backend

## Requirements

- Python 3.11+ and Node.js 20+
- A free **Groq API key**: https://console.groq.com/keys
- ~3 GB of disk space. No GPU needed.

## Run it

```bash
git clone https://github.com/HIJOdelIDANII/5G-RAG.git
cd 5G-RAG

make setup                  # install Python + frontend dependencies
cp .env.example .env        # then put your key in .env:  LLM_API_KEY_GROK=gsk_...
make data                   # download the two 3GPP specs and parse them (~1 min)
make index                  # build the search indexes (one time, ~30–45 min on CPU)
```

Then start the app in **two terminals**:

```bash
make api                    # terminal 1: backend, wait for "Application startup complete"
make web                    # terminal 2: open http://localhost:5173
```

<details>
<summary>Without <code>make</code> (e.g. on Windows)</summary>

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd web && npm install && cd ..

python -m ingest.download_specs
python -m ingest.parse_docx
python -m ingest.build_index

uvicorn api.server:app --port 8000        # terminal 1
cd web && npm run dev                      # terminal 2: http://localhost:5173
```
</details>

## Project structure

```
ingest/         download the specs, parse DOCX into chunks, build the indexes
rag/            retriever (routing, hybrid search, fusion, reranker) and generator (prompt, LLM, citations)
api/server.py   FastAPI backend
web/            React + TypeScript frontend
```
