import httpx
import respx

from regintel.embeddings.ollama_embedder import OllamaEmbedder


@respx.mock
def test_embed_batch_returns_vectors():
    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    )
    emb = OllamaEmbedder(host="http://localhost:11434", model="bge-m3")
    out = emb.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
def test_embed_one():
    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.5, 0.6]]})
    )
    emb = OllamaEmbedder(host="http://localhost:11434", model="bge-m3")
    assert emb.embed_one("hello") == [0.5, 0.6]
