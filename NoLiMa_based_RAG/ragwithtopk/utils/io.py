# ragwithtopk/utils/io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional
import hashlib

def load_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_lines(path: str | Path) -> List[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
