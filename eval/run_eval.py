"""Evaluate retrieval and generation on eval/testset.json.

Usage:
    python -m eval.run_eval                 # retrieval ablations + generation  -> eval/results/
    python -m eval.run_eval --retrieval     # retrieval only (no LLM calls)

Retrieval (in-corpus questions): a question is a hit at k if ANY expected section is in the top k.
    recall@k for k in 1, 3, 5, 10 and MRR@10, for 5 configurations (full pipeline and ablations).
Generation (full pipeline, all questions):
    refusal accuracy (out-of-corpus refused, in-corpus answered), citation validity (cited sections
    are among the retrieved passages), citation correctness (at least one expected section cited),
    and an LLM judge (a different model from the generator) for faithfulness and correctness.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from rag.generator import Generator
from rag.retriever import Retriever

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TESTSET = HERE / "testset.json"
KS = (1, 3, 5, 10)
JUDGE_MODEL = "qwen/qwen3.8-27b"   # different family from the generator (gpt-oss-120b): less self-bias

CONFIGS = {
    "Vector only":            dict(hybrid=False, rerank=False, route_query=False),
    "Full − hybrid (no BM25)": dict(hybrid=False, rerank=True, route_query=True),
    "Full − reranker":         dict(hybrid=True, rerank=False, route_query=True),
    "Full − routing":          dict(hybrid=True, rerank=True, route_query=False),
    "Full pipeline":           dict(hybrid=True, rerank=True, route_query=True),
}


def first_hit_rank(cites: list[str], expected: set[str]) -> int | None:
    for i, c in enumerate(cites, 1):
        if c in expected:
            return i
    return None


def eval_retrieval(retriever: Retriever, questions: list[dict]) -> dict:
    out = {}
    for name, cfg in CONFIGS.items():
        rows, t0 = [], time.perf_counter()
        for q in questions:
            r = retriever.retrieve(q["question"], k=10, **cfg)
            cites = [h.citation for h in r.hits]
            rank = first_hit_rank(cites, set(q["expected"]))
            specs_top5 = {h.chunk["spec"] for h in r.hits[:5]}
            rows.append({"id": q["id"], "type": q["type"], "rank": rank, "top5": cites[:5],
                         "route": r.specs, "both_specs_top5": len(specs_top5) == 2})
        n = len(rows)
        m = {f"recall@{k}": sum(1 for x in rows if x["rank"] and x["rank"] <= k) / n for k in KS}
        m["MRR@10"] = sum(1 / x["rank"] for x in rows if x["rank"]) / n
        for t in ("simple", "cross"):
            sub = [x for x in rows if x["type"] == t]
            m[f"recall@5 ({t})"] = sum(1 for x in sub if x["rank"] and x["rank"] <= 5) / len(sub)
        cross = [x for x in rows if x["type"] == "cross"]
        m["cross: both specs in top 5"] = sum(x["both_specs_top5"] for x in cross) / len(cross)
        m["sec/question"] = (time.perf_counter() - t0) / n
        out[name] = {"metrics": m, "rows": rows}
        print(f"{name:<26} " + "  ".join(f"{k}={v:.2f}" for k, v in m.items()), flush=True)
    return out


JUDGE_PROMPT = """You grade an answer produced by a RAG chatbot about 3GPP 5G specifications.

QUESTION: {question}

RETRIEVED PASSAGES (the only allowed source):
{passages}

ANSWER TO GRADE:
{answer}

REFERENCE (key facts a correct answer should contain):
{reference}

Grade two things:
- faithfulness: 1 if every factual statement in the ANSWER is supported by the RETRIEVED PASSAGES, else 0.
  (An explicit statement that something is not covered is faithful.)
- correctness: 1 if the ANSWER contains the key facts of the REFERENCE, 0.5 if partially, 0 if not or wrong.

Reply with ONLY a JSON object: {{"faithfulness": 0 or 1, "correctness": 0, 0.5 or 1, "comment": "<one short sentence>"}}"""


def judge(client, q: dict, a) -> dict:
    passages = "\n\n".join(f"[{h.citation}] {h.chunk['text'][:1500]}" for h in a.retrieval.hits)
    msg = JUDGE_PROMPT.format(question=q["question"], passages=passages, answer=a.text,
                              reference=q["reference"])
    for _ in range(3):
        resp = client.chat.completions.create(model=JUDGE_MODEL, temperature=0, max_tokens=2000,
                                              messages=[{"role": "user", "content": msg}])
        text = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.S)
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"faithfulness": None, "correctness": None, "comment": "judge parse error"}


def eval_generation(gen: Generator, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        a = gen.answer(q["question"])
        expected = set(q["expected"])
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "refused": a.refused,
               "citations": a.citations, "invalid": a.invalid_citations,
               "cites_expected": bool(expected & set(a.citations)), "answer": a.text,
               "latency": sum(a.timings.values())}
        if q["type"] != "out" and not a.refused:
            row.update(judge(gen.client, q, a))
        rows.append(row)
        print(f"{q['id']:<7} refused={a.refused!s:<5} cites={len(a.citations)} invalid={len(a.invalid_citations)} "
              f"expected_cited={row['cites_expected']!s:<5} faith={row.get('faithfulness')} "
              f"correct={row.get('correctness')}", flush=True)

    ins = [r for r in rows if r["type"] != "out"]
    outs = [r for r in rows if r["type"] == "out"]
    answered = [r for r in ins if not r["refused"]]
    all_cites = sum(len(r["citations"]) for r in answered)
    judged = [r for r in answered if r.get("faithfulness") is not None]
    m = {
        "out-of-corpus correctly refused": sum(r["refused"] for r in outs) / len(outs),
        "in-corpus wrongly refused": sum(r["refused"] for r in ins) / len(ins),
        "citation validity (cited ∈ retrieved)": 1 - sum(len(r["invalid"]) for r in answered) / max(all_cites, 1),
        "answers citing an expected section": sum(r["cites_expected"] for r in answered) / max(len(answered), 1),
        "faithfulness (judge)": sum(r["faithfulness"] for r in judged) / max(len(judged), 1),
        "correctness (judge)": sum(r["correctness"] for r in judged) / max(len(judged), 1),
        "avg latency (s)": sum(r["latency"] for r in rows) / len(rows),
    }
    for t in ("simple", "cross"):
        sub = [r for r in judged if r["type"] == t]
        if sub:
            m[f"correctness (judge, {t})"] = sum(r["correctness"] for r in sub) / len(sub)
    return {"metrics": m, "rows": rows}


def write_report(ret: dict | None, gen: dict | None) -> None:
    lines = ["# Evaluation results", "", f"Test set: `eval/testset.json` (see counts below). Generated by `python -m eval.run_eval`.", ""]
    if ret:
        lines += ["## Retrieval (26 in-corpus questions)", "",
                  "| Configuration | R@1 | R@3 | R@5 | R@10 | MRR@10 | R@5 simple | R@5 cross | cross: both specs in top 5 | s/question |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in ret.items():
            m = d["metrics"]
            lines.append(f"| {name} | {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} | {m['recall@10']:.2f} | "
                         f"{m['MRR@10']:.2f} | {m['recall@5 (simple)']:.2f} | {m['recall@5 (cross)']:.2f} | "
                         f"{m['cross: both specs in top 5']:.2f} | {m['sec/question']:.1f} |")
        lines += ["", "Misses of the full pipeline (no expected section in the top 10):", ""]
        for r in ret["Full pipeline"]["rows"]:
            if not r["rank"]:
                lines.append(f"- {r['id']}: got {', '.join(r['top5'])}")
        lines.append("")
    if gen:
        lines += ["## Generation (full pipeline, 29 questions)", "", "| Metric | Value |", "|---|---|"]
        for k, v in gen["metrics"].items():
            lines.append(f"| {k} | {v:.2f} |")
        lines += ["", f"Judge model: `{JUDGE_MODEL}` (different from the generator, `openai/gpt-oss-120b`).", "",
                  "| id | refused | citations (invalid) | expected cited | faithful | correct | judge comment |",
                  "|---|---|---|---|---|---|---|"]
        for r in gen["rows"]:
            lines.append(f"| {r['id']} | {r['refused']} | {len(r['citations'])} ({len(r['invalid'])}) | {r['cites_expected']} | "
                         f"{r.get('faithfulness', '–')} | {r.get('correctness', '–')} | {str(r.get('comment', '')).replace('|', '/')} |")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true", help="retrieval metrics only")
    args = ap.parse_args()

    questions = json.loads(TESTSET.read_text())
    RESULTS.mkdir(exist_ok=True)
    retriever = Retriever()
    ret = eval_retrieval(retriever, [q for q in questions if q["type"] != "out"])
    (RESULTS / "retrieval.json").write_text(json.dumps(ret, indent=2, ensure_ascii=False))
    gen = None
    if not args.retrieval:
        gen = eval_generation(Generator(retriever=retriever), questions)
        (RESULTS / "generation.json").write_text(json.dumps(gen, indent=2, ensure_ascii=False))
    write_report(ret, gen)
    print(f"report: {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
