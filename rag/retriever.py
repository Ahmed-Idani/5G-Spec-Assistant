"""Online retrieval: question -> top-k chunks from the 3GPP corpus.

Pipeline position (runs on every chat message):
    question -> [routing] -> [vector search + BM25] -> [RRF fusion] -> [reranker] -> top-k -> LLM

    1. Routing   : rule-based classifier picks TS 23.501, TS 24.501 or both, and filters the search.
    2. Search    : vector search (bge-base-en-v1.5 in Chroma) and BM25 keyword search, N candidates each.
    3. Fusion    : Reciprocal Rank Fusion, score = sum 1 / (RRF_K + rank) over both lists.
    4. Reranking : cross-encoder (bge-reranker-base) rescores (question, chunk) pairs and keeps the top k.

Every stage can be switched off (hybrid / rerank / route) for the evaluation comparisons.

CLI:
    python -m rag.retriever "What is timer T3510 used for?"
    python -m rag.retriever "..." --no-hybrid --no-rerank --no-route -k 5
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ingest.build_index import (BM25_FILE, CHROMA_DIR, COLLECTION, EMBED_MODEL,
                                QUERY_INSTRUCTION, bm25_tokenize, load_chunks)

RERANK_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60            # standard RRF constant: damps the weight of top ranks
N_CANDIDATES = 20     # per search method, before fusion
MAX_PER_SECTION = 2   # diversity: at most this many chunks (parts) of the same section in the top k
SPECS = ("TS 23.501", "TS 24.501")

# ---------------------------------------------------------------- routing
# TS 24.501 = NAS signalling (Stage 3): messages, IEs, causes, timers, states, coding.
NAS_PATTERNS = [
    r"\bnas\b", r"\b5gmm\b", r"\b5gsm\b", r"\bcause\b", r"#\d+", r"\bt3[0-9]{3}\b",
    r"\btimer", r"\bmessage", r"\binformation element", r"\bie\b", r"\boctet", r"\bcod(e|ing)\b",
    r"\b(request|accept|reject|complete|command)\b", r"\bstate\b", r"\b5gmm-[a-z-]+",
    r"\bsecurity mode\b", r"\bderegistration\b", r"\babnormal case", r"\bmessage type",
    r"\bue policy\b", r"\bsor\b", r"\bpdu session (establishment|modification|release) (request|accept|reject)",
]
# TS 23.501 = system architecture (Stage 2): NFs, reference points, concepts.
ARCH_PATTERNS = [
    r"\barchitecture\b", r"\bnetwork functions?\b", r"\brole\b", r"\bfunctionalit", r"\breference point",
    r"\bservice[- ]based\b", r"\binterface\b", r"\bn\d{1,2}\b", r"\b(smf|upf|pcf|udm|ausf|nrf|nssf|nef|udr|scp|sepp|nwdaf|af|chf)\b",
    r"\bqos (model|flow|characteristics)", r"\b5qi\b", r"\bslic", r"\bs-nssai\b", r"\broaming\b",
    r"\bselection\b", r"\bdeploy", r"\bnon-public network|\bnpn\b|\bsnpn\b", r"\bedge computing",
    r"\bnetwork slice", r"\buser plane\b", r"\bcontrol plane\b", r"\bconcept",
    # procedures and the AMF are described in both specs: they pull cross-spec questions to "both"
    r"\bprocedure", r"\bamf\b",
]
EXPLICIT_SPEC = {"TS 23.501": r"23\.?501", "TS 24.501": r"24\.?501"}


def route(question: str) -> tuple[list[str], dict]:
    """Return (specs to search, debug info). Two specs = no filter."""
    q = question.lower()
    explicit = [s for s, p in EXPLICIT_SPEC.items() if re.search(p, q)]
    if explicit:
        return explicit, {"reason": "explicit spec in question"}
    nas = sum(bool(re.search(p, q)) for p in NAS_PATTERNS)
    arch = sum(bool(re.search(p, q)) for p in ARCH_PATTERNS)
    info = {"nas_score": nas, "arch_score": arch}
    # One side clearly dominant -> that spec only; otherwise (tie, both, none) -> both.
    if nas >= 1 and arch == 0 or nas >= arch + 2:
        return ["TS 24.501"], {**info, "reason": "NAS signals"}
    if arch >= 1 and nas == 0 or arch >= nas + 2:
        return ["TS 23.501"], {**info, "reason": "architecture signals"}
    return list(SPECS), {**info, "reason": "mixed or no signal -> both specs"}


# ---------------------------------------------------------------- retriever
@dataclass
class Hit:
    chunk: dict
    score: float                       # final score (rerank score, or RRF, or vector similarity)
    ranks: dict = field(default_factory=dict)   # rank in each stage, for debugging / the UI

    @property
    def citation(self) -> str:
        return f"{self.chunk['spec']} §{self.chunk['section']}"


@dataclass
class RetrievalResult:
    question: str
    specs: list[str]
    route_info: dict
    hits: list[Hit]
    timings: dict


class Retriever:
    """Loads the models and indexes once; call .retrieve() per question."""

    def __init__(self, load_reranker: bool = True):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.chunks = {c["chunk_id"]: c for c in load_chunks()}
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
        with BM25_FILE.open("rb") as f:
            data = pickle.load(f)
        self.bm25, self.bm25_ids = data["bm25"], data["ids"]
        self.bm25_spec = np.array([self.chunks[i]["spec"] for i in self.bm25_ids])
        self.reranker = None
        if load_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=512)

    # -- individual search methods -------------------------------------------
    def vector_search(self, question: str, specs: list[str], n: int) -> list[tuple[str, float]]:
        vec = self.embedder.encode(QUERY_INSTRUCTION + question, normalize_embeddings=True).tolist()
        where = {"spec": specs[0]} if len(specs) == 1 else None
        res = self.collection.query(query_embeddings=[vec], n_results=n, where=where,
                                    include=["distances"])
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]  # cosine sim

    def bm25_search(self, question: str, specs: list[str], n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(bm25_tokenize(question))
        if len(specs) == 1:
            scores = np.where(self.bm25_spec == specs[0], scores, -np.inf)
        top = np.argsort(-scores)[:n]
        return [(self.bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    @staticmethod
    def rrf(*ranked_lists: list[tuple[str, float]]) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for lst in ranked_lists:
            for rank, (cid, _) in enumerate(lst, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda x: -x[1])

    # -- full pipeline ----------------------------------------------------------
    def retrieve(self, question: str, k: int = 5, hybrid: bool = True, rerank: bool = True,
                 route_query: bool = True, n_candidates: int = N_CANDIDATES) -> RetrievalResult:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        specs, info = route(question) if route_query else (list(SPECS), {"reason": "routing disabled"})

        vec = self.vector_search(question, specs, n_candidates)
        t["vector"] = time.perf_counter() - t0
        ranks: dict[str, dict] = {cid: {"vector": r} for r, (cid, _) in enumerate(vec, 1)}

        if hybrid:
            t1 = time.perf_counter()
            bm = self.bm25_search(question, specs, n_candidates)
            t["bm25"] = time.perf_counter() - t1
            for r, (cid, _) in enumerate(bm, 1):
                ranks.setdefault(cid, {})["bm25"] = r
            candidates = self.rrf(vec, bm)
        else:
            candidates = vec

        if rerank and self.reranker is not None:
            t1 = time.perf_counter()
            pool = candidates[:n_candidates]
            scores = self.reranker.predict([(question, self.chunks[cid]["text"]) for cid, _ in pool])
            candidates = sorted(zip([c for c, _ in pool], map(float, scores)), key=lambda x: -x[1])
            t["rerank"] = time.perf_counter() - t1

        hits, per_section = [], {}
        for cid, score in candidates:
            c = self.chunks[cid]
            key = (c["spec"], c["section"])
            if per_section.get(key, 0) >= MAX_PER_SECTION:
                continue
            per_section[key] = per_section.get(key, 0) + 1
            ranks.setdefault(cid, {})["final"] = len(hits) + 1
            hits.append(Hit(c, score, ranks[cid]))
            if len(hits) == k:
                break
        t["total"] = time.perf_counter() - t0
        return RetrievalResult(question, specs, info, hits, t)


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the 3GPP retriever.")
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-hybrid", action="store_true", help="vector search only")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-route", action="store_true", help="always search both specs")
    args = ap.parse_args()

    r = Retriever(load_reranker=not args.no_rerank).retrieve(
        args.question, k=args.k, hybrid=not args.no_hybrid,
        rerank=not args.no_rerank, route_query=not args.no_route)
    print(f"Q: {r.question}\nRoute: {r.specs} ({r.route_info})")
    print("Timings: " + ", ".join(f"{k}={v:.2f}s" for k, v in r.timings.items()))
    for i, h in enumerate(r.hits, 1):
        print(f"{i}. [{h.score:7.3f}] {h.citation:<22} {h.chunk['title'][:60]}  ranks={h.ranks}")


if __name__ == "__main__":
    main()
