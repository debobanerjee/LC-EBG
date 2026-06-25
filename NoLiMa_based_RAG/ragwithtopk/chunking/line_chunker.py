# ragwithtopk/chunking/line_chunker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ragwithtopk.utils.io import read_lines

@dataclass
class LineChunker:
    drop_empty: bool = True

    def chunk_file(self, path: str) -> List[str]:
        lines = read_lines(path)
        if self.drop_empty:
            lines = [x for x in lines if x.strip()]
        return lines
