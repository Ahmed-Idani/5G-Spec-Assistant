"""Build the search indexes from corpus/chunks.jsonl.

Usage:
    python -m ingest.build_index

Pipeline position: chunks.jsonl -> [THIS] -> index/chroma (vectors) + index/bm25.pkl (keywords)

  1. Vector index: embeds every chunk with an English embedding model (bge-base-en-v1.5,
     768 dims, normalized) and stores vectors + metadata in a persistent ChromaDB collection
     (cosine distance). Metadata allows filtering by spec, which query routing uses.
  2. Keyword index: BM25 (Okapi) over a tokenizer that keeps 3GPP identifiers intact
     (T3510, 5GMM, 9.11.3.2, N11, PDU ...).
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "corpus" / "chunks.jsonl"
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_FILE = INDEX_DIR / "bm25.pkl"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION = "spec_chunks"
# bge v1.5: queries get this instruction, passages are embedded as-is.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

META_FIELDS = ("spec", "version", "release", "source_file", "section", "title",
               "parent_section", "section_path", "level", "part", "n_parts")

STOPWORDS = set("""a an and are as at be by for from has have if in into is it its of on or
that the their then there these this to was were what when which while with does do how
why who shall may can not no""".split())
# '#9' is kept with its '#': in 3GPP text "#<n>" is a cause value, so it must not
# collide with every other number 9 in the corpus.
TOKEN_RE = re.compile(r"#\d+|[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase, keep dotted/hyphenated identifiers and '#<n>' causes as one token, drop stopwords.

    "Cause #9 in 5GMM (see 9.11.3.2), timer T3510" ->
    ['cause', '#9', '5gmm', 'see', '9.11.3.2', 'timer', 't3510']
    """
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_vector_index(chunks: list[dict]) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    print(f"Embedded {len(texts)} chunks in {time.time() - t0:.0f}s, dim={vectors.shape[1]}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)  # rebuild from scratch
    col = client.create_collection(COLLECTION, metadata={
        "hnsw:space": "cosine", "embed_model": EMBED_MODEL})

    batch = 1000
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=vectors[i:i + batch].tolist(),
            documents=[c["text"] for c in part],
            # Chroma metadata values cannot be None
            metadatas=[{k: ("" if c[k] is None else c[k]) for k in META_FIELDS} for c in part],
        )
    print(f"Chroma collection '{COLLECTION}': {col.count()} vectors -> {CHROMA_DIR}")


def build_bm25_index(chunks: list[dict]) -> None:
    from rank_bm25 import BM25Okapi

    tokenized = [bm25_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with BM25_FILE.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["chunk_id"] for c in chunks]}, f)
    print(f"BM25 index: {len(chunks)} docs, vocab {len(bm25.idf)} terms -> {BM25_FILE}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build BM25 and vector indexes.")
    ap.add_argument("--bm25-only", action="store_true",
                    help="rebuild only the BM25 index (seconds; vectors take ~30-45 min on CPU)")
    args = ap.parse_args()

    chunks = load_chunks()
    INDEX_DIR.mkdir(exist_ok=True)
    build_bm25_index(chunks)
    if not args.bm25_only:
        build_vector_index(chunks)


if __name__ == "__main__":
    main()
