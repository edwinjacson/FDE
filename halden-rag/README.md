# Halden RAG: question answering over messy policy documents

## The customer problem

A mid-sized logistics company keeps its policies in ten different places and formats: a 2024 handbook and a 2025 update that overrides part of it, a policy PDF scanned through bad OCR, a Slack-era FAQ with stale answers, and meeting notes containing a decision that was never written back into the policy. Staff keep asking HR, Finance and IT the same questions.

The goal is an assistant that answers from the documents, **cites its sources**, **prefers newer information when documents conflict**, and **says "I don't know" instead of guessing**. Just as important: a way to *measure* whether it does those things, so changes can be judged by numbers rather than by vibes.

> The documents are synthetic (written to contain realistic traps), so the answers are known and the evaluation is repeatable.

## What's in the box

| File | Purpose |
|------|---------|
| `docs/` | 10 documents: handbook + update, expenses, remote work, IT security, incident runbook, vendors, OCR-noisy travel policy, messy meeting notes, stale FAQ |
| `rag.py` | Chunking, two retrievers (TF-IDF keyword, local embeddings), generation with citations |
| `eval.py` | Evaluation harness with retrieval and answer metrics, saved runs, comparison table |
| `ask.py` | Interactive Q&A |
| `eval/questions.jsonl` | 24 questions: lookups, a table, OCR noise, conflicting documents, multi-document, and 4 unanswerable |

## Setup (Windows)

Requires Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your-key      (or set it in your PyCharm run configuration)
```

## Run

```
python eval.py --retrieval-only        # no API key, no cost: measures retrieval alone
python eval.py                         # full eval with answers from Claude
python ask.py                          # try your own questions
python eval.py --compare               # table of every saved run
```

## Experiments (fill in the results as you run them)

Change one thing at a time, then record the numbers.

| # | Question | Command | hit@k | answer acc | refusal | in tokens |
|---|----------|---------|-------|-----------|---------|-----------|
| 1 | Baseline | `python eval.py` | | | | |
| 2 | Does fewer retrieved chunks hurt? | `python eval.py --top-k 1` | | | | |
| 3 | Do smaller chunks help? | `python eval.py --chunk-size 300 --overlap 50` | | | | |
| 4 | Does a filename header help? | `python eval.py --chunk-size 300 --overlap 50 --header` | | | | |
| 5 | Semantic vs keyword retrieval | `python eval.py --backend embed` | | | | |
| 6 | Semantic with small chunks, few results | `python eval.py --backend embed --chunk-size 300 --top-k 2` | | | | |

Embeddings need `pip install fastembed` (downloads a small model on first run).

## Findings

_Write this section last, from your own results. Suggested structure:_

1. **What worked.** Which setting gave the best accuracy for the lowest token cost?
2. **What failed and why.** Pick 2-3 failures from the report and explain the cause (retrieval miss vs the model ignoring evidence vs a genuinely conflicting source).
3. **The conflict problem.** How does the system behave when a newer decision contradicts the formal policy? What would you change in the data process, not just the code?
4. **Refusals.** Does it say "I don't know" when it should? What happens on the maternity-pay question, where the topic is mentioned but the answer isn't?
5. **What I'd do for a real customer.** Metadata (dates, versions) in the index, hybrid retrieval, a larger eval set built from real questions, monitoring of live failures, and a review workflow for answers about money or legal matters.

## Known limitations

- Ten short documents is far smaller than a real corpus, so retrieval looks easier than it would be in production.
- Answer scoring uses keyword matching, which is strict but can reject a correct answer phrased differently. Check failures by hand before trusting the number.
- Questions were written by the same person who wrote the documents. Real users ask differently.
