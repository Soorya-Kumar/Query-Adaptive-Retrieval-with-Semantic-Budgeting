import httpx
from typing import List

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"  # good balance for 7b-class hardware, change if needed


def embed(text: str) -> List[float]:
    response = httpx.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: List[str]) -> List[List[float]]:
    return [embed(t) for t in texts]