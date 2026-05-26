import json
from hashlib import sha256
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        return [round(digest[index % len(digest)] / 255, 6) for index in range(self.dimensions)]


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._client = client or OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            params: dict[str, Any] = {"model": self._model, "input": batch}
            if self._dimensions is not None:
                params["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**params)
            embeddings.extend([list(row.embedding) for row in response.data])
        return embeddings


class CachedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, cache_path: Path) -> None:
        self._provider = provider
        self._cache_path = cache_path
        self._lock = Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        ordered_unique = list(dict.fromkeys(texts))
        with self._lock:
            cache = self._read_cache()
        missing = [text for text in ordered_unique if self._cache_key(text) not in cache]
        if missing:
            vectors = self._provider.embed(missing)
            with self._lock:
                latest_cache = self._read_cache()
                for text, vector in zip(missing, vectors, strict=True):
                    latest_cache.setdefault(self._cache_key(text), vector)
                self._write_cache(latest_cache)
                cache = latest_cache
        return [cache[self._cache_key(text)] for text in texts]

    def _read_cache(self) -> dict[str, list[float]]:
        if not self._cache_path.exists():
            return {}
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            corrupt_path = self._cache_path.with_suffix(
                self._cache_path.suffix + f".{get_ident()}.corrupt"
            )
            try:
                self._cache_path.replace(corrupt_path)
            except OSError:
                pass
            return {}
        return {str(key): [float(value) for value in vector] for key, vector in payload.items()}

    def _write_cache(self, cache: dict[str, list[float]]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(cache, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self._cache_path)

    @staticmethod
    def _cache_key(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()
