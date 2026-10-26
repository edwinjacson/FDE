"""Core RAG pipeline: load -> chunk -> retrieve -> generate with citations.

Kept deliberately small and framework-free so you can explain every line.

Retrieval backends (same `search(query, k)` interface):
  tfidf   keyword baseline, no downloads
  embed   dense embeddings held in a NumPy matrix (in memory, rebuilt every run)
  chroma  the same embeddings stored in a persistent Chroma vector database,
          with metadata (source, document date) that can be filtered on
"""
import argparse
import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import numpy as np

MODEL = "claude-haiku-5-5"
ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
INDEX_DIR = ROOT / "index"  # Chroma stores its database here
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions about Halden Logistics internal policies using ONLY the documents provided.

Rules:
1. Use only the provided documents. Do not use outside knowledge.
2. Cite every factual claim with its source id in the form [source: <id>], using the ids shown on the documents.
3. Documents can conflict or be out of date. Prefer the most recently dated document or decision (document dates are shown when available), and briefly mention the conflict.
4. If the answer is not in the documents, reply exactly: {REFUSAL}
5. Be concise."""


# ----------------------------------------------------------------------------
# Loading and chunking
# ----------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str            # e.g. "expenses_policy.md#2"
    source: str        # file name
    text: str
    updated: str = ""  # ISO date taken from the document header; "" if unknown

    @property
    def updated_num(self) -> int:
        """Date as a number like 20250318 (0 = unknown). Chroma filters on numbers, not strings."""
        return int(self.updated.replace("-", "")) if self.updated else 0


DATE_LINE = re.compile(r"^(?:last updated|last edited|date):\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_doc_date(text: str) -> str:
    """Find a 'Last updated: 3 June 2024' style line. Returns ISO date or '' (e.g. OCR-garbled dates)."""
    m = DATE_LINE.search(text)
    if not m:
        return ""
    for fmt in ("%d %B %Y", "%B %Y"):
        try:
            return datetime.strptime(m.group(1), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


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
        updated = parse_doc_date(text)
        for i, part in enumerate(chunk_text(text, size, overlap)):
            body = f"[{name}]\n{part}" if header else part  # header = cheap "contextual retrieval"
            chunks.append(Chunk(id=f"{name}#{i}", source=name, text=body, updated=updated))
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


class FastEmbedder:
    """Small local embedding model (fastembed, ONNX, no torch). Returns L2-normalised vectors."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)  # downloads the model on first use

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class EmbedRetriever:
    """Dense retrieval with the vectors held in a NumPy matrix."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedder = FastEmbedder()
        self.matrix = self.embedder.embed([c.text for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q  # cosine similarity (both sides normalised)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class ChromaRetriever:
    """Same embeddings as EmbedRetriever, stored in a persistent Chroma vector database.

    What this adds over the NumPy version:
      * persistence: chunks are embedded once and the index is reused on later runs
      * metadata: each chunk stores its source file and document date, so queries can
        filter (see --after)
      * approximate-nearest-neighbour search that scales far beyond what a NumPy matrix can
    The collection name is a hash of the model and chunk contents, so changing the
    chunk size, overlap or header automatically builds a fresh index.
    """

    def __init__(self, chunks: list[Chunk], after: str | None = None):
        import chromadb

        self.by_id = {c.id: c for c in chunks}
        self.embedder = FastEmbedder()
        self.where = {"updated": {"$gte": int(after.replace("-", ""))}} if after else None

        digest = hashlib.sha1((EMBED_MODEL + "".join(c.id + c.text for c in chunks)).encode("utf-8"))
        name = f"halden-{digest.hexdigest()[:12]}"
        client = chromadb.PersistentClient(path=str(INDEX_DIR))

        try:
            self.collection = client.get_collection(name)
            if self.collection.count() != len(chunks):
                raise ValueError("stale or partial index")
            print(f"Chroma: reusing index {name} ({len(chunks)} chunks)")
        except Exception:
            try:
                client.delete_collection(name)
            except Exception:
                pass
            self.collection = client.create_collection(name)  # default distance: squared L2
            self.collection.add(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=self.embedder.embed([c.text for c in chunks]).tolist(),
                metadatas=[{"source": c.source, "updated": c.updated_num} for c in chunks],
            )
            print(f"Chroma: built index {name} ({len(chunks)} chunks) in {INDEX_DIR.name}/")

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])[0]
        res = self.collection.query(
            query_embeddings=[q.tolist()],
            n_results=min(k, self.collection.count()),
            where=self.where,  # metadata filter, or None
        )
        # Vectors are unit length, so squared L2 distance d = 2 - 2*cosine. Convert back to similarity.
        return [(self.by_id[cid], 1.0 - dist / 2.0) for cid, dist in zip(res["ids"][0], res["distances"][0])]


def make_retriever(backend: str, chunks: list[Chunk], after: str | None = None):
    if after and backend != "chroma":
        raise SystemExit("--after needs --backend chroma (it relies on Chroma's metadata filtering).")
    if backend == "tfidf":
        return TfidfRetriever(chunks)
    if backend == "embed":
        return EmbedRetriever(chunks)
    if backend == "chroma":
        return ChromaRetriever(chunks, after)
    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def format_context(results: list[tuple[Chunk, float]], dates: bool = False) -> str:
    """Wrap chunks in <document> tags. With dates=True the tag also carries the document date."""
    parts = []
    for c, _ in results:
        date_attr = f' updated="{c.updated or "unknown"}"' if dates else ""
        parts.append(f'<document id="{c.id}"{date_attr}>\n{c.text}\n</document>')
    return "<documents>\n" + "\n".join(parts) + "\n</documents>"


def get_client():
    import anthropic  # imported lazily so retrieval-only runs need no API key
    load_dotenv()
    return anthropic.Anthropic()


def generate(client, question: str, results: list[tuple[Chunk, float]], dates: bool = False) -> tuple[str, dict]:
    prompt = f"{format_context(results, dates)}\n\nQuestion: {question}"
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
    parser.add_argument("--backend", choices=["tfidf", "embed", "chroma"], default="tfidf")
    parser.add_argument("--chunk-size", type=int, default=800, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="characters repeated between chunks")
    parser.add_argument("--top-k", type=int, default=4, help="chunks passed to the model")
    parser.add_argument("--header", action="store_true", help="prefix each chunk with its file name")
    parser.add_argument("--dates", action="store_true",
                        help="show each document's date to the model so it can resolve conflicts")
    parser.add_argument("--after", metavar="YYYY-MM-DD",
                        help="chroma only: retrieve only from documents dated on/after this day")


def build_pipeline(args) -> tuple[list[Chunk], object]:
    docs = load_documents()
    chunks = build_chunks(docs, args.chunk_size, args.overlap, args.header)
    return chunks, make_retriever(args.backend, chunks, args.after)
