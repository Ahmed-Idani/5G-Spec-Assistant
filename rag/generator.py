"""Grounded answer generation: question + retrieved chunks -> English answer with citations.

Pipeline position (online, after retrieval):
    question -> Retriever (top-k chunks) -> [THIS: prompt + LLM] -> answer + [TS 24.501 §A.1] citations

Guarantees aimed at (project requirements):
  - English answers, based only on the retrieved context (no outside knowledge).
  - Every claim cites its source as [TS xx.xxx §section].
  - Explicit refusal "I could not find this information in the documents." when the context
    does not contain the answer. Two layers:
      1. Pre-LLM: if the best reranker score is below REFUSAL_THRESHOLD, refuse without calling the LLM.
      2. In-prompt: the LLM is instructed to refuse when the context is insufficient.
  - Post-check: every citation in the answer is verified against the retrieved chunks;
    citations that point to sections not in the context are reported as invalid.

LLM: openai/gpt-oss-120b served by Groq (OpenAI-compatible API). Key: LLM_API_KEY_GROK in .env.

CLI:
    python -m rag.generator "What does cause #9 mean in 5GMM?"
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from ingest.build_index import ROOT
from rag.retriever import Hit, RetrievalResult, Retriever

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
REFUSAL = "I could not find this information in the documents."
# Reranker score below which no retrieved chunk is considered relevant (out-of-corpus question).
# Observed: out-of-corpus ~0.000, in-corpus questions 0.1-0.99. To be tuned on the test set.
REFUSAL_THRESHOLD = 0.02

SYSTEM_PROMPT = f"""You are a 5G standards assistant. You answer questions about the 3GPP specifications \
TS 23.501 (5G System architecture, Stage 2) and TS 24.501 (NAS protocol for 5GS, Stage 3), Release 18.

STRICT RULES:
1. Use ONLY the information in the CONTEXT passages below. Do not use prior knowledge, even if you \
believe it is correct. Do not invent message names, cause values, timer values, section numbers or procedures.
2. Answer in English, whatever the language of the question.
3. Cite the source after every statement, using the exact label of the passage it comes from, \
in plain square brackets with normal spaces, exactly as written, e.g. [TS 24.501 §5.5.1.2] \
(no other bracket style, no sub-labels such as (a) or (1)). Only cite labels that appear in the CONTEXT. \
When a statement combines several passages, cite all of them.
4. If the CONTEXT does not contain the information needed to answer, reply exactly: "{REFUSAL}" \
and nothing else. If it answers only part of the question, answer that part and say clearly which part \
is not covered by the documents.
5. Be precise and technical. Keep 3GPP terminology (message names in capitals, IE names, cause numbers). \
Use short paragraphs or bullet points; for procedures, give the steps in order.
"""


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        c = h.chunk
        body = c["text"].split("\n\n", 1)[-1]  # drop the retrieval header line, the label replaces it
        blocks.append(f"[{h.citation}] {c['title']} (path: {c['section_path']})\n{body}")
    return "\n\n---\n\n".join(blocks)


# Tolerant: any Unicode space (the model often emits U+202F), [ ] or 【 】 brackets, "§" optional.
CITATION_RE = re.compile(r"TS\s*(2[34]\.501)\s*§?\s*((?:Annex\s+)?[A-Za-z]?\.?\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*(?:~\d+)?)")


def extract_citations(answer: str) -> list[str]:
    """'... [TS 24.501 §A.1] ...' -> ['TS 24.501 §A.1'] (unique, in order)."""
    seen = []
    for num, sec in CITATION_RE.findall(answer):
        cite = f"TS {num} §{re.sub(r'\\s+', ' ', sec)}"
        if cite not in seen:
            seen.append(cite)
    return seen


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    refusal_reason: str | None
    citations: list[str]                    # citations found in the answer
    invalid_citations: list[str]            # cited, but not among the retrieved chunks
    retrieval: RetrievalResult
    model: str
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    reranked: bool = True                   # whether retrieval scores are reranker scores


class Generator:
    def __init__(self, retriever: Retriever | None = None, model: str = DEFAULT_MODEL):
        load_dotenv(ROOT / ".env")
        key = os.getenv("LLM_API_KEY_GROK")
        if not key:
            raise RuntimeError("LLM_API_KEY_GROK missing: copy .env.example to .env and set it")
        self.client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
        self.model = model
        self.retriever = retriever or Retriever()

    def answer(self, question: str, k: int = 5, threshold: float = REFUSAL_THRESHOLD,
               **retrieve_kwargs) -> Answer:
        """Retrieve + generate in one call."""
        r = self.retriever.retrieve(question, k=k, **retrieve_kwargs)
        return self.generate(r, threshold=threshold, reranked=retrieve_kwargs.get("rerank", True))

    def generate(self, r: RetrievalResult, threshold: float = REFUSAL_THRESHOLD,
                 reranked: bool = True) -> Answer:
        """Generate the answer from an existing retrieval result (lets the UI show each stage)."""
        question = r.question
        timings = dict(retrieval=r.timings["total"])

        # Layer 1: nothing relevant retrieved -> refuse without calling the LLM.
        # Only meaningful with the reranker (its scores are calibrated relevance in [0, 1]).
        if not r.hits or (reranked and r.hits[0].score < threshold):
            return Answer(question, REFUSAL, True, f"best rerank score below {threshold}",
                          [], [], r, self.model, timings)

        user = (f"CONTEXT:\n\n{format_context(r.hits)}\n\n=====\n\n"
                f"QUESTION: {question}\n\nAnswer following the rules, with citations.")
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=1500,
        )
        timings["llm"] = time.perf_counter() - t0
        text = (resp.choices[0].message.content or "").strip()

        # Layer 2: the LLM itself refused.
        refused = REFUSAL.lower().rstrip(".") in text.lower() and len(text) < len(REFUSAL) + 40
        cites = extract_citations(text)
        allowed = {h.citation for h in r.hits}
        invalid = [c for c in cites if c not in allowed]
        usage = resp.usage.model_dump() if resp.usage else {}
        return Answer(question, text, refused, "LLM: context insufficient" if refused else None,
                      cites, invalid, r, self.model, timings, usage)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the 3GPP RAG chatbot.")
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    a = Generator(model=args.model).answer(args.question, k=args.k)
    print(f"Q: {a.question}\nRoute: {a.retrieval.specs}\n")
    print(a.text)
    print("\nSources retrieved:")
    for h in a.retrieval.hits:
        print(f"  [{h.score:.3f}] {h.citation:<22} {h.chunk['title'][:60]}")
    print(f"\nCitations used: {a.citations}  invalid: {a.invalid_citations}  refused: {a.refused}")
    print("Timings: " + ", ".join(f"{k}={v:.1f}s" for k, v in a.timings.items()))


if __name__ == "__main__":
    main()
