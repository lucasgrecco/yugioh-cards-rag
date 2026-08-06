"""Shared embedding utilities.

Provides a single, reusable interface for generating text embeddings.
Supports both OpenAI API and local models via sentence-transformers.
Auto-detects the provider based on whether OPENAI_API_KEY is set.
"""

import logging

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_LOCAL_MODEL,
    EMBEDDING_PROVIDER,
)

logger = logging.getLogger(__name__)

# OpenAI client (lazy init)
_openai_client: OpenAI | None = None

# Local model (lazy init)
_local_model: SentenceTransformer | None = None


def _get_openai_client() -> OpenAI:
    """Return a lazily-initialized OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Create a .env file with your API key."
            )
        _openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            max_retries=3,
            timeout=30.0,
        )
    return _openai_client


def _get_local_model() -> SentenceTransformer:
    """Return a lazily-initialized local SentenceTransformer model.

    Uses the GPU when torch sees one and falls back to CPU otherwise, following the
    same device selection as the reranker in `retrieval.py`. A missing GPU is slow,
    not fatal: the machine without one still needs to run ingest and search.

    `"cuda"` is also the right device string on a ROCm build of torch — HIP presents
    itself through the CUDA API, so an AMD card needs no branch of its own here, only
    the ROCm wheel at install time.
    """
    global _local_model
    if _local_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading local embedding model: %s (device=%s)",
            EMBEDDING_LOCAL_MODEL,
            device,
        )
        _local_model = SentenceTransformer(EMBEDDING_LOCAL_MODEL, device=device)
        try:
            _local_model.encode("test")
        except RuntimeError:
            logger.warning("GPU unusable (%s), falling back to CPU", device)
            device = "cpu"
            _local_model = SentenceTransformer(
                EMBEDDING_LOCAL_MODEL, device=device
            )
        logger.info("Local model loaded successfully on %s", device)
    return _local_model


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _get_openai_embedding(text: str) -> list[float]:
    """Generate an embedding via OpenAI API."""
    logger.debug("Generating OpenAI embedding for text (%d chars)", len(text))
    response = _get_openai_client().embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def _get_local_embedding(text: str) -> list[float]:
    """Generate an embedding via local SentenceTransformer model."""
    logger.debug("Generating local embedding for text (%d chars)", len(text))
    model = _get_local_model()
    return model.encode(text, normalize_embeddings=True).tolist()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _get_openai_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate batch embeddings via OpenAI API."""
    logger.debug("Generating OpenAI batch embeddings for %d texts", len(texts))
    response = _get_openai_client().embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def _get_local_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate batch embeddings via local SentenceTransformer model."""
    logger.debug("Generating local batch embeddings for %d texts", len(texts))
    model = _get_local_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Uses OpenAI if OPENAI_API_KEY is set, otherwise uses a local
    SentenceTransformer model.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If using OpenAI and OPENAI_API_KEY is not set.
        openai.APIError: If the OpenAI API request fails.
    """
    if EMBEDDING_PROVIDER == "openai":
        return _get_openai_embedding(text)
    return _get_local_embedding(text)


def get_both_embeddings(
    text: str,
) -> tuple[list[float] | None, list[float] | None]:
    """Generate both OpenAI and local embeddings for a single text.

    Always generates the local embedding. Generates OpenAI embedding
    only if OPENAI_API_KEY is set.

    Returns:
        Tuple of (openai_vector, local_vector). Each is None if
        that provider is unavailable.
    """
    openai_vec = None
    if OPENAI_API_KEY:
        openai_vec = _get_openai_embedding(text)
    local_vec = _get_local_embedding(text)
    return openai_vec, local_vec


def get_both_embeddings_batch(
    texts: list[str],
) -> tuple[list[list[float]] | None, list[list[float]] | None]:
    """Generate both OpenAI and local embeddings for multiple texts.

    Always generates local embeddings. Generates OpenAI embeddings
    only if OPENAI_API_KEY is set.

    Returns:
        Tuple of (openai_vectors, local_vectors). Each is None if
        that provider is unavailable.
    """
    openai_vecs = None
    if OPENAI_API_KEY:
        openai_vecs = _get_openai_embeddings_batch(texts)
    local_vecs = _get_local_embeddings_batch(texts)
    return openai_vecs, local_vecs


def get_local_embedding(text: str) -> list[float]:
    """Generate a local embedding vector for the given text.

    Always uses the local SentenceTransformer model (mxbai-embed-large-v1)
    on CUDA, regardless of OPENAI_API_KEY. Used by the MCP server which
    always runs local embeddings.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    return _get_local_embedding(text)
