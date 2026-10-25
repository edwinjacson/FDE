"""Core RAG pipeline: load -> chunk -> retrieve -> generate with citations.

Kept deliberately small and framework-free so you can explain every line.
"""
import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

import numpy as np

MODEL = "claude-haiku-5-5"
DOCS_DIR = Path(__file__).parent / "docs"
REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions about Halden Logistics internal policies using ONLY the documents provided.

Rules:
1. Use only the provided documents. Do not use outside knowledge.
2. Cite every factual claim with its source id in the form [source: <id>], using the ids shown on the documents.
3. Documents can conflict or be out of date. Prefer the most recently dated document or decision, and briefly mention the conflict.
4. If the answer is not in the documents, reply exactly: {REFUSAL}
5. Be concise."""


# ----------------------------------------------------------------------------
# Loading and chunking
# ----------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str      # e.g. "expenses_policy.md#2"
    source: str  # file name
    text: str


def load_documents(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    return {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(docs_dir.glob("*"))
        if p.suffix in {".md", ".txt"}
    }


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of at most ~`size` characters.

    Paragraph boundaries are respected where possible; paragraphs longer than
    `size` are split on sentence boundaries. `overlap` characters from the end of
    the previous chunk are repeated at the start of the next one (trimmed to a
    word boundary) so facts at a boundary aren't lost.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) > size:
            pieces.extend(s for s in re.split(r"(?<=[.!?])\s+", p) if s)
        else:
            pieces.append(p)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if tail else ""  # start on a word boundary
            current = tail
        current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        chunks.append(current)
    return chunks


def build_chunks(docs: dict[str, str], size: int, overlap: int, header: bool = False) -> list[Chunk]:
    chunks = []
    for name, text in docs.items():
        for i, part in enumerate(chunk_text(text, size, overlap)):
            body = f"[{name}]\n{part}" if header else part  # header = cheap "contextual retrieval"
            chunks.append(Chunk(id=f"{name}#{i}", source=name, text=body))
    return chunks


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------
class TfidfRetriever:
    """Keyword baseline. No API, no downloads: build this first."""

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class EmbedRetriever:
    """Semantic retrieval with a small local embedding model (via fastembed, ONNX, no torch)."""

    def __init__(self, chunks: list[Chunk], model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.chunks = chunks
        self.model = TextEmbedding(model_name)  # downloads the model on first use
        vectors = np.array(list(self.model.embed([c.text for c in chunks])))
        self.matrix = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = np.array(list(self.model.embed([query])))[0]
        q = q / np.linalg.norm(q)
        scores = self.matrix @ q  # cosine similarity (both sides normalised)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def make_retriever(backend: str, chunks: list[Chunk]):
    if backend == "tfidf":
        return TfidfRetriever(chunks)
    if backend == "embed":
        return EmbedRetriever(chunks)
    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def format_context(results: list[tuple[Chunk, float]]) -> str:
    docs = "\n".join(f'<document id="{c.id}">\n{c.text}\n</document>' for c, _ in results)
    return f"<documents>\n{docs}\n</documents>"


def get_client():
    import anthropic  # imported lazily so retrieval-only runs need no API key
    load_dotenv()
    return anthropic.Anthropic()


def generate(client, question: str, results: list[tuple[Chunk, float]]) -> tuple[str, dict]:
    prompt = f"{format_context(results)}\n\nQuestion: {question}"
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,  # thinking tokens count toward this limit
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    seconds = time.perf_counter() - t0
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "seconds": seconds}
    return text, usage


def extract_citations(answer: str) -> list[str]:
    """Pull ids out of [source: a.md#0, b.md#1] style citations."""
    ids: list[str] = []
    for group in re.findall(r"\[source:\s*([^\]]+)\]", answer):
        ids.extend(i.strip() for i in group.split(",") if i.strip())
    return ids


def is_refusal(answer: str) -> bool:
    return "don't know based on" in answer.lower().replace("\u2019", "'")


# ----------------------------------------------------------------------------
# Shared CLI options (used by eval.py and ask.py)
# ----------------------------------------------------------------------------
def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["tfidf", "embed"], default="tfidf")
    parser.add_argument("--chunk-size", type=int, default=800, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="characters repeated between chunks")
    parser.add_argument("--top-k", type=int, default=4, help="chunks passed to the model")
    parser.add_argument("--header", action="store_true", help="prefix each chunk with its file name")


def build_pipeline(args) -> tuple[list[Chunk], object]:
    docs = load_documents()
    chunks = build_chunks(docs, args.chunk_size, args.overlap, args.header)
    return chunks, make_retriever(args.backend, chunks)
