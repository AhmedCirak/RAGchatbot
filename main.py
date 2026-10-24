"""
main.py - FastAPI backend for the RAG chatbot.

Run:
    uvicorn main:app --reload
Then open http://localhost:8000/docs
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import Iterator, Literal

import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

import rag
from config import BASE_DIR, LLM_MODEL, LLM_PROVIDER

STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("rag-api")


# ---------- Schemas (request/response shapes, validated automatically) ----------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["What is this document about?"])
    history: list[Message] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class Source(BaseModel):
    source: str
    page: int
    similarity: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    provider: str
    model: str
    document: str | None = None
    chunks: int | None = None
    detail: str | None = None


# ---------- Startup: load models once, before the first request ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag.get_collection()
        list(rag.get_embedder().query_embed(["warm-up"]))  # loads the ONNX model into memory
        log.info("Index and embedding model loaded. Ready.")
    except Exception as e:
        # Don't crash the server - /health will report the problem.
        log.error("Startup check failed: %s", e)
    yield


app = FastAPI(
    title="RAG Chatbot API",
    version="0.1.0",
    description="Ask questions about a document. Answers are grounded in the document and cite pages.",
    lifespan=lifespan,
)

# Lets a frontend on another address call this API (e.g. while developing).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------- Error handling ----------

def to_http_error(e: Exception) -> HTTPException:
    """Maps internal / LLM provider errors to clean HTTP errors."""
    if isinstance(e, openai.RateLimitError):
        return HTTPException(429, "The language model is rate limited. Try again in a moment.")
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return HTTPException(502, "Could not reach the language model provider.")
    if isinstance(e, openai.APIStatusError):
        return HTTPException(502, f"Language model error: {e.message}")
    if isinstance(e, RuntimeError):  # our own config/index errors from rag.py
        return HTTPException(503, str(e))
    log.exception("Unexpected error")
    return HTTPException(500, "Internal server error.")


def history_as_dicts(request: ChatRequest) -> list[dict]:
    return [m.model_dump() for m in request.history]


# ---------- Endpoints ----------

@app.get("/", include_in_schema=False)
def root():
    """Serves the chat web page (falls back to the API docs if it's missing)."""
    page = STATIC_DIR / "index.html"
    return FileResponse(page) if page.exists() else RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    try:
        collection = rag.get_collection()
        first = collection.get(limit=1, include=["metadatas"])["metadatas"]
        document = first[0]["source"] if first else None
        return HealthResponse(
            status="ok", provider=LLM_PROVIDER, model=LLM_MODEL,
            document=document, chunks=collection.count(),
        )
    except Exception as e:
        raise HTTPException(
            503,
            HealthResponse(status="error", provider=LLM_PROVIDER, model=LLM_MODEL, detail=str(e)).model_dump(),
        )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """Returns the whole answer at once (used e.g. by the Telegram bot)."""
    log.info("Q: %s", request.question)
    try:
        return rag.answer(request.question, history_as_dicts(request))
    except Exception as e:
        raise to_http_error(e)


def sse(event: str, data: dict) -> str:
    """Formats one Server-Sent Event. Data is JSON, so newlines in tokens are safe."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream", tags=["chat"])
def chat_stream(request: ChatRequest):
    """
    Streams the answer as Server-Sent Events:
    - `sources` - once, before the answer: {"sources": [...]}
    - `token`   - many times: {"text": "..."}
    - `error`   - if generation fails midway: {"detail": "..."}
    - `done`    - once, at the end: {}
    """
    log.info("Q (stream): %s", request.question)
    try:
        # Retrieval happens here, so its errors still return a normal HTTP error.
        sources, tokens = rag.answer_stream(request.question, history_as_dicts(request))
    except Exception as e:
        raise to_http_error(e)

    def events() -> Iterator[str]:
        yield sse("sources", {"sources": sources})
        try:
            for token in tokens:
                yield sse("token", {"text": token})
        except Exception as e:
            # Headers are already sent, so the error has to go inside the stream.
            yield sse("error", {"detail": to_http_error(e).detail})
        yield sse("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
