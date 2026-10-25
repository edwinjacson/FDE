"""Evaluation harness.

Retrieval metrics need NO API key:
    python eval.py --retrieval-only

Full run (retrieval + answers from Claude):
    python eval.py
    python eval.py --backend embed --top-k 6 --header

Compare all saved runs:
    python eval.py --compare

Metrics
  hit@k      gold document appears in the top-k retrieved chunks (answerable questions)
  MRR        1 / rank of the first gold document (answerable questions)
  answer_acc every `must_include` group is present in the answer (a|b = alternatives)
  refusal    on unanswerable questions, the model says it doesn't know
  cite_ok    answer cites at least one chunk, and every cited id was actually retrieved
  cite_gold  at least one citation points at a gold document
"""
import argparse
import json
import time
from pathlib import Path

from rag import add_pipeline_args, build_pipeline, extract_citations, generate, get_client, is_refusal

ROOT = Path(__file__).parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def terms_present(answer: str, groups: list[str]) -> bool:
    a = answer.lower()
    return all(any(alt.lower() in a for alt in group.split("|")) for group in groups)


def first_gold_rank(retrieved, gold_docs) -> int | None:
    for rank, (chunk, _) in enumerate(retrieved, start=1):
        if chunk.source in gold_docs:
            return rank
    return None


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def run_eval(args) -> None:
    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    chunks, retriever = build_pipeline(args)
    print(f"{len(chunks)} chunks | backend={args.backend} size={args.chunk_size} "
          f"overlap={args.overlap} top_k={args.top_k} header={args.header}\n")

    client = None if args.retrieval_only else get_client()
    rows = []

    for q in questions:
        retrieved = retriever.search(q["question"], args.top_k)
        row = {"id": q["id"], "category": q["category"], "question": q["question"],
               "retrieved": [c.id for c, _ in retrieved], "should_refuse": q["should_refuse"]}

        if not q["should_refuse"]:
            rank = first_gold_rank(retrieved, q["gold_docs"])
            row["hit"] = rank is not None
            row["rr"] = 1 / rank if rank else 0.0

        if client:
            answer, usage = generate(client, q["question"], retrieved)
            cites = extract_citations(answer)
            row.update(answer=answer, usage=usage, citations=cites)
            if q["should_refuse"]:
                row["correct"] = is_refusal(answer)
            else:
                row["correct"] = (not is_refusal(answer)) and terms_present(answer, q["must_include"])
                retrieved_ids = {c.id for c, _ in retrieved}
                row["cite_ok"] = bool(cites) and all(c in retrieved_ids for c in cites)
                row["cite_gold"] = any(c.split("#")[0] in q["gold_docs"] for c in cites)
        rows.append(row)

    summary = summarise(rows, has_answers=client is not None)
    print_report(rows, summary, has_answers=client is not None)
    save_run(args, rows, summary)


def summarise(rows, has_answers: bool) -> dict:
    answerable = [r for r in rows if not r["should_refuse"]]
    unanswerable = [r for r in rows if r["should_refuse"]]
    s = {"hit_at_k": mean(r["hit"] for r in answerable), "mrr": mean(r["rr"] for r in answerable)}
    if has_answers:
        s.update(
            answer_acc=mean(r["correct"] for r in answerable),
            refusal_acc=mean(r["correct"] for r in unanswerable),
            cite_ok=mean(r["cite_ok"] for r in answerable),
            cite_gold=mean(r["cite_gold"] for r in answerable),
            avg_input_tokens=mean(r["usage"]["input_tokens"] for r in rows),
            avg_output_tokens=mean(r["usage"]["output_tokens"] for r in rows),
            avg_seconds=mean(r["usage"]["seconds"] for r in rows),
        )
    return s


def print_report(rows, summary, has_answers: bool) -> None:
    print("=" * 72)
    for k, v in summary.items():
        print(f"{k:>18}: {v:.2f}" if isinstance(v, float) else f"{k:>18}: {v}")
    print("=" * 72)

    # Per-category breakdown: this is where the story usually is.
    cats = sorted({r["category"] for r in rows})
    print("\nBy category" + ("  (hit@k / answer correct)" if has_answers else "  (hit@k)"))
    for cat in cats:
        group = [r for r in rows if r["category"] == cat]
        hits = [r["hit"] for r in group if "hit" in r]
        line = f"  {cat:<13} n={len(group):<2}"
        if hits:
            line += f" hit={mean(hits):.2f}"
        if has_answers:
            line += f" correct={mean(r['correct'] for r in group):.2f}"
        print(line)

    print("\nFailures")
    failed = 0
    for r in rows:
        retrieval_miss = r.get("hit") is False
        wrong = has_answers and not r["correct"]
        if not (retrieval_miss or wrong):
            continue
        failed += 1
        why = []
        if retrieval_miss:
            why.append("RETRIEVAL MISS")
        if wrong:
            why.append("WRONG ANSWER")
        print(f"- {r['id']} [{r['category']}] {r['question']}  -> {', '.join(why)}")
        print(f"    retrieved: {r['retrieved']}")
        if has_answers:
            print(f"    answer: {r['answer'][:220].strip()!r}")
    if not failed:
        print("  none")


def save_run(args, rows, summary) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    config = {k: getattr(args, k) for k in ("backend", "chunk_size", "overlap", "top_k", "header")}
    config["retrieval_only"] = args.retrieval_only
    name = (f"{time.strftime('%Y%m%d-%H%M%S')}__{args.backend}_c{args.chunk_size}_o{args.overlap}"
            f"_k{args.top_k}{'_hdr' if args.header else ''}{'_retonly' if args.retrieval_only else ''}.json")
    (RESULTS_DIR / name).write_text(json.dumps({"config": config, "summary": summary, "rows": rows},
                                               indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results/{name}")


def compare_runs() -> None:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No saved runs yet.")
        return
    print(f"{'run':<58}{'hit@k':>7}{'mrr':>6}{'ans':>6}{'refuse':>8}{'cite':>6}{'in_tok':>8}")
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        s = d["summary"]
        fmt = lambda key, w: f"{s[key]:>{w}.2f}" if key in s else f"{'-':>{w}}"
        print(f"{f.stem.split("__", 1)[1]:<58}{fmt('hit_at_k', 7)}{fmt('mrr', 6)}{fmt('answer_acc', 6)}"
              f"{fmt('refusal_acc', 8)}{fmt('cite_ok', 6)}{fmt('avg_input_tokens', 8)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_pipeline_args(parser)
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM; no API key needed")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--compare", action="store_true", help="print a table of all saved runs and exit")
    args = parser.parse_args()
    compare_runs() if args.compare else run_eval(args)
