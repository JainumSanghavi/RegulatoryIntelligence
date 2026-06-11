from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str


def _token_len(text: str) -> int:
    return len(_ENC.encode(text))


def chunk_text(
    text: str,
    *,
    doc_id: str,
    chunk_tokens: int = 800,
    overlap_tokens: int = 150,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p for p in splitter.split_text(text) if p.strip()]
    if not pieces:
        return []
    return [Chunk(doc_id=doc_id, chunk_index=i, text=p) for i, p in enumerate(pieces)]
