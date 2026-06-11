from dataclasses import dataclass


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    """Sparse lexical vectors via FastEmbed BM25 (non-neural)."""

    def __init__(self, model=None, model_name: str = "Qdrant/bm25") -> None:
        if model is not None:
            self._model = model
        else:
            from fastembed import SparseTextEmbedding
            self._model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> list[SparseVec]:
        out: list[SparseVec] = []
        for emb in self._model.embed(texts):
            out.append(SparseVec(indices=list(emb.indices), values=list(emb.values)))
        return out

    def encode_query(self, text: str) -> SparseVec:
        if hasattr(self._model, "query_embed"):
            emb = next(iter(self._model.query_embed(text)))
        else:
            emb = next(iter(self._model.embed([text])))
        return SparseVec(indices=list(emb.indices), values=list(emb.values))
