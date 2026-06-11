from dataclasses import dataclass
from pathlib import Path


@dataclass
class InternalDoc:
    doc_id: str
    title: str
    text: str
    doc_type: str
    source: str = "internal"
    jurisdiction: str = "internal"


def _infer_doc_type(stem: str) -> str:
    if "sop" in stem:
        return "sop"
    if "contract" in stem or "agreement" in stem:
        return "contract"
    return "policy"


def load_internal_docs(directory: Path) -> list[InternalDoc]:
    docs: list[InternalDoc] = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        docs.append(
            InternalDoc(
                doc_id=path.stem,
                title=title,
                text=text,
                doc_type=_infer_doc_type(path.stem),
            )
        )
    return docs
