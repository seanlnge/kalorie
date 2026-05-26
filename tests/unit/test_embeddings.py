from pathlib import Path

from kalorie.ml.embeddings import (
    CachedEmbeddingProvider,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


class FakeOpenAIEmbeddings:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "EmbeddingResponse",
            (),
            {
                "data": [
                    type("EmbeddingData", (), {"embedding": [0.1, 0.2]})(),
                    type("EmbeddingData", (), {"embedding": [0.3, 0.4]})(),
                ]
            },
        )()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeOpenAIEmbeddings()


def test_fake_embedding_provider_is_deterministic():
    provider = FakeEmbeddingProvider(dimensions=4)

    first = provider.embed(["traffic", "same restaurant sales"])
    second = provider.embed(["traffic", "same restaurant sales"])

    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 4 for vector in first)


def test_openai_embedding_provider_uses_official_client_boundary():
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(
        client=client,
        model="text-embedding-3-large",
        dimensions=256,
    )

    embeddings = provider.embed(["traffic", "margin"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-3-large",
            "input": ["traffic", "margin"],
            "dimensions": 256,
        }
    ]


def test_cached_embedding_provider_reuses_cached_vectors(tmp_path: Path):
    provider = FakeEmbeddingProvider(dimensions=3)
    cached = CachedEmbeddingProvider(provider=provider, cache_path=tmp_path / "embeddings.json")

    first = cached.embed(["traffic", "traffic"])
    second = cached.embed(["traffic"])

    assert first == [second[0], second[0]]
    assert (tmp_path / "embeddings.json").exists()
