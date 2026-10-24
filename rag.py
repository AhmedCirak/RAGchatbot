"""
rag.py - retrieval + generation over the indexed document.

The FastAPI backend will import `answer()` and `answer_stream()` from here.
Run it directly for an interactive chat in the terminal:
    python rag.py
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import chromadb
from fastembed import TextEmbedding
from openai import OpenAI

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_HISTORY_TURNS,
    MIN_SIMILARITY,
    OLLAMA_BASE_URL,
    TEMPERATURE,
    TOP_K,
)

NOT_FOUND_ANSWER = "I couldn't find information about that in the document."

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a single document.

Rules:
- Answer ONLY using the context excerpts provided. Do not use outside knowledge.
- If the context does not contain the answer, say you couldn't find it in the document. Never guess.
- Cite the pages you used in square brackets, e.g. [p. 4] or [p. 4, 7].
- Be concise and clear. Use short paragraphs or bullet points when it helps."""

# A message in the chat history: {"role": "user" | "assistant", "content": "..."}
Message = dict[str, str]


@dataclass
class Chunk:
    text: str
    page: int
    source: str
    similarity: float


# ---------- Lazy-loaded resources (created once, reused for every question) ----------

@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError("Index not found. Run `python ingest.py` first.") from e
    if collection.count() == 0:
        raise RuntimeError("Index is empty. Run `python ingest.py` first.")
    return collection


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    # Groq and Ollama both expose an OpenAI-compatible API, so one client covers both.
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    if LLM_PROVIDER == "ollama":
        return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)  # key is required but ignored
    raise RuntimeError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' (use 'groq' or 'ollama').")


# ---------- Retrieval ----------

def retrieve(question: str, k: int = TOP_K, min_similarity: float = MIN_SIMILARITY) -> list[Chunk]:
    collection = get_collection()
    query_embedding = next(iter(get_embedder().query_embed([question]))).tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
    )

    chunks = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        similarity = 1 - distance  # collection uses cosine distance
        if similarity >= min_similarity:
            chunks.append(Chunk(text, int(meta["page"]), meta["source"], round(similarity, 3)))
    return chunks


def format_sources(chunks: list[Chunk]) -> list[dict]:
    """One entry per page (best-matching chunk), sorted by page - this is what the UI shows."""
    best: dict[int, dict] = {}
    for c in chunks:
        if c.page not in best or c.similarity > best[c.page]["similarity"]:
            best[c.page] = {
                "source": c.source,
                "page": c.page,
                "similarity": c.similarity,
                "snippet": c.text[:200] + ("..." if len(c.text) > 200 else ""),
            }
    return sorted(best.values(), key=lambda s: s["page"])


# ---------- Generation ----------

def build_messages(question: str, chunks: list[Chunk], history: list[Message] | None = None) -> list[Message]:
    context = "\n\n".join(f"[p. {c.page}]\n{c.text}" for c in chunks)
    messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Only the plain previous Q&A goes into history (not their context), to keep the prompt short.
    for turn in (history or [])[-MAX_HISTORY_TURNS * 2:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Context from the document:\n\n{context}\n\n---\nQuestion: {question}",
    })
    return messages


def answer(question: str, history: list[Message] | None = None) -> dict:
    """Full answer at once: {"answer": str, "sources": [...]}"""
    chunks = retrieve(question)
    if not chunks:  # nothing relevant -> don't even call the LLM
        return {"answer": NOT_FOUND_ANSWER, "sources": []}

    response = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=build_messages(question, chunks, history),
        temperature=TEMPERATURE,
    )
    return {"answer": response.choices[0].message.content.strip(), "sources": format_sources(chunks)}


def answer_stream(question: str, history: list[Message] | None = None) -> tuple[list[dict], Iterator[str]]:
    """Returns (sources, token generator). Sources are known before generation starts,
    so the UI can show them right away while the answer is still being typed out."""
    chunks = retrieve(question)
    if not chunks:
        return [], iter([NOT_FOUND_ANSWER])

    def tokens() -> Iterator[str]:
        stream = get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=build_messages(question, chunks, history),
            temperature=TEMPERATURE,
            stream=True,
        )
        for event in stream:
            if event.choices and (delta := event.choices[0].delta.content):
                yield delta

    return format_sources(chunks), tokens()


# ---------- Terminal chat ----------

def main() -> None:
    print(f"RAG chat  |  provider: {LLM_PROVIDER}  |  model: {LLM_MODEL}")
    print("Type your question ('exit' to quit).\n")
    history: list[Message] = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            sources, tokens = answer_stream(question, history)
            print("Bot: ", end="", flush=True)
            reply = ""
            for token in tokens:
                print(token, end="", flush=True)
                reply += token
            print()
            if sources:
                pages = ", ".join(f"p. {s['page']} ({s['similarity']:.2f})" for s in sources)
                print(f"     Sources: {pages}")
            print()
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        except RuntimeError as e:
            print(f"\nError: {e}\n")
            break
        except Exception as e:  # network / API errors - keep the chat alive
            print(f"\nError while answering: {e}\n")

    print("Bye!")


if __name__ == "__main__":
    main()
