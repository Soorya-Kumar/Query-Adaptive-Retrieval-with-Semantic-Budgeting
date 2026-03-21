import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    doc_id: str
    raw_text: str
    metadata: dict  # filename, path, size, etc.


def load_txt(path: str) -> Document:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    return Document(
        doc_id=p.stem,
        raw_text=text,
        metadata={
            "filename": p.name,
            "path": str(p.resolve()),
            "size_bytes": p.stat().st_size,
        }
    )


def load_directory(dir_path: str) -> List[Document]:
    docs = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".txt"):
            docs.append(load_txt(os.path.join(dir_path, fname)))
    return docs