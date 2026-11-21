"""HTTP API for the 3GPP RAG chatbot (used by the React frontend in web/).

Run:
    .venv/bin/uvicorn api.server:app --port 8000

Endpoints:
    GET  /api/config   models, defaults, corpus info
    POST /api/ask      streams NDJSON events, one JSON object per line:
                         {"type": "route",     "specs": [...], "reason": "..."}
                         {"type": "retrieval", "hits": [...], "timings": {...}}
                         {"type": "answer",    "text": "...", "refused": bool, "citations": [...], ...}
                         {"type": "error",     "message": "..."}
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.generator import DEFAULT_MODEL, REFUSAL_THRESHOLD, Generator
from rag.retriever import SPECS, Hit, Retriever, route

MODELS = [
    {"id": "openai/gpt-oss-120b", "label": "GPT-OSS 120B"},
    {"id": "openai/gpt-oss-20b", "label": "GPT-OSS 20B"},
    {"id": "qwen/qwen3.8-27b", "label": "Qwen 3.8 27B"},
]
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["retriever"] = Retriever(load_reranker=True)  # load models + indexes once (~10 s)
    state["generators"] = {}
    yield


app = FastAPI(title="5G 3GPP RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"],
                   allow_headers=["*"])


def generator(model: str) -> Generator:
    gens = state["generators"]
    if model not in gens:
        gens[model] = Generator(retriever=state["retriever"], model=model)
    return gens[model]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str = DEFAULT_MODEL
    k: int = Field(5, ge=1, le=10)
    hybrid: bool = True
    rerank: bool = True
    route: bool = True
    threshold: float = Field(REFUSAL_THRESHOLD, ge=0, le=1)


def hit_json(h: Hit) -> dict:
    c = h.chunk
    return {
        "id": c["chunk_id"], "citation": h.citation, "spec": c["spec"], "section": c["section"],
        "title": c["title"], "path": c["section_path"], "part": c["part"], "n_parts": c["n_parts"],
        "score": h.score, "ranks": h.ranks, "text": c["text"].split("\n\n", 1)[-1],
    }


@app.get("/api/config")
def config() -> dict:
    return {
        "models": MODELS, "default_model": DEFAULT_MODEL, "threshold": REFUSAL_THRESHOLD,
        "corpus": [{"spec": "TS 23.501", "version": "18.12.0", "title": "System architecture for the 5G System"},
                   {"spec": "TS 24.501", "version": "18.14.0", "title": "NAS protocol for 5GS; Stage 3"}],
        "chunks": len(state["retriever"].chunks),
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    if req.model not in {m["id"] for m in MODELS}:
        req.model = DEFAULT_MODEL

    def events():
        def emit(obj: dict) -> str:
            return json.dumps(obj, ensure_ascii=False) + "\n"
        try:
            specs, info = route(req.question) if req.route else (list(SPECS), {"reason": "routing disabled"})
            yield emit({"type": "route", "specs": specs, "reason": info["reason"]})

            r = state["retriever"].retrieve(req.question, k=req.k, hybrid=req.hybrid,
                                            rerank=req.rerank, route_query=req.route)
            yield emit({"type": "retrieval", "hits": [hit_json(h) for h in r.hits],
                        "timings": r.timings, "reranked": req.rerank})

            a = generator(req.model).generate(r, threshold=req.threshold, reranked=req.rerank)
            yield emit({"type": "answer", "text": a.text, "refused": a.refused,
                        "refusal_reason": a.refusal_reason, "citations": a.citations,
                        "invalid_citations": a.invalid_citations, "model": a.model,
                        "timings": a.timings, "usage": a.usage})
        except Exception as e:  # surface API / network errors to the UI
            yield emit({"type": "error", "message": str(e)})

    # sync generator -> FastAPI runs it in a worker thread, so the blocking model calls are fine
    return StreamingResponse(events(), media_type="application/x-ndjson")
