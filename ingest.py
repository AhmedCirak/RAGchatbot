"""
ingest.py - indexes a single PDF into a local Chroma vector store.

Usage:
    python ingest.py                                  # uses data/document.pdf
    python ingest.py --pdf path/to/file.pdf
    python ingest.py --query "What is this document about?"   # ingest, then test retrieval
"""
import argparse
import re
import sys
import time
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from fastembed import TextEmbedding

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_PDF,
    EMBEDDING_MODEL,
    MIN_CHUNK_CHARS,
)

BATCH_SIZE = 256


# ---------- 1. Extraction ----------

def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")                  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)                # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)             # max one blank line
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single line breaks -> space (keeps paragraphs)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns [(page_number, text), ...], 1-based page numbers, empty pages skipped."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((page_number, text))
    return pages


# ---------- 2. Chunking ----------

def _split_long(piece: str, size: int) -> list[str]:
    """Splits a paragraph that is too long: first by sentences, then hard-cuts as a last resort."""
    if len(piece) <= size:
        return [piece]
    parts = []
    for sentence in re.split(r"(?<=[.!?])\s+", piece):
        if len(sentence) <= size:
            parts.append(sentence)
        else:
            parts.extend(sentence[i:i + size] for i in range(0, len(sentence), size))
    return parts


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Packs paragraphs/sentences into chunks of up to `size` chars, with overlap between chunks."""
    pieces = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            pieces.extend(_split_long(paragraph, size))

    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        if " " in tail:                       # start the overlap at a word boundary
            tail = tail.split(" ", 1)[1]
        with_tail = f"{tail} {piece}".strip()
        current = with_tail if len(with_tail) <= size else piece  # never exceed `size`
    if current:
        chunks.append(current)
    return chunks


def build_chunks(pages: list[tuple[int, str]], source: str) -> list[dict]:
    """Chunks each page separately so every chunk maps to exactly one page (for citations)."""
    chunks = []
    for page_number, text in pages:
        for i, chunk_text in enumerate(split_text(text)):
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "id": f"p{page_number}-c{i}",
                "text": chunk_text,
                "metadata": {"source": source, "page": page_number, "chunk": i},
            })
    return chunks


# ---------- 3. Embedding + storage ----------

def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(chunks: list[dict], model: TextEmbedding) -> None:
    texts = [c["text"] for c in chunks]
    # passage_embed = embeddings for documents; questions later use query_embed
    embeddings = [e.tolist() for e in model.passage_embed(texts, batch_size=32)]

    collection = get_collection(reset=True)  # re-running ingest rebuilds the index from scratch
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embeddings[start:start + BATCH_SIZE],
        )


def test_query(question: str, model: TextEmbedding, k: int = 3) -> None:
    collection = get_collection()
    query_embedding = next(iter(model.query_embed([question]))).tolist()
    result = collection.query(query_embeddings=[query_embedding], n_results=k)

    print(f'\nTop {k} chunks for: "{question}"\n')
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        print(f"[page {meta['page']}]  similarity: {1 - dist:.3f}")
        print(f"  {doc[:300]}{'...' if len(doc) > 300 else ''}\n")


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF into Chroma.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Path to the PDF")
    parser.add_argument("--query", type=str, help="Optional test question after ingesting")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    t0 = time.perf_counter()
    pages = extract_pages(args.pdf)
    if not pages:
        sys.exit("No text extracted. Is the PDF scanned? (it would need OCR)")
    print(f"Extracted text from {len(pages)} pages")

    chunks = build_chunks(pages, source=args.pdf.name)
    avg = sum(len(c["text"]) for c in chunks) // len(chunks)
    print(f"Created {len(chunks)} chunks (avg {avg} chars)")

    print(f"Loading embedding model {EMBEDDING_MODEL} (first run downloads it)...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    store_chunks(chunks, model)
    print(f"Stored in {CHROMA_DIR} in {time.perf_counter() - t0:.1f}s")

    if args.query:
        test_query(args.query, model)


if __name__ == "__main__":
    main()
