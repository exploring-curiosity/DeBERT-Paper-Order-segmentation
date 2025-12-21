from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Set


def iter_pdf_paths(root: str | Path, *, recursive: bool = True) -> Iterable[Path]:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() == ".pdf":
        yield root_path
        return

    pattern = "**/*.pdf" if recursive else "*.pdf"
    for p in sorted(root_path.glob(pattern)):
        if p.is_file() and p.suffix.lower() == ".pdf":
            yield p


def list_pdf_paths(root: str | Path, *, recursive: bool = True) -> List[Path]:
    return list(iter_pdf_paths(root, recursive=recursive))


def filter_pdfs_by_name(
    pdf_paths: Iterable[Path],
    *,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> List[Path]:
    include = {s for s in (include or set())}
    exclude = {s for s in (exclude or set())}

    out: List[Path] = []
    for p in pdf_paths:
        name = p.name
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue
        out.append(p)
    return out
