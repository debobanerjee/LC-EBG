# ragwithtopk/injection/inserter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import random
import os
from pathlib import Path

ChunkType = Literal["needle", "document"]

@dataclass
class NeedleInserter:
    allow_before_first: bool = True

    def inject(
        self,
        base_chunks: List[str],
        needles: List[Dict[str, Any]],
        *,
        seed: int,
        source_file: str,
        run_id: str,
    ) -> List[Dict[str, Any]]:
        """
        base_chunks: document chunks (line-based)
        needles: list dicts from NeedleExtractor.extract(...)
        returns: list of chunk records with sequential chunk_id starting at 1
        """
        rng = random.Random(seed)

        n_lines = len(base_chunks)
        n_needles = len(needles)
        source = self._get_source_name(source_file)
        run_id = f"{run_id}|{source}"
        if n_needles == 0:
            return [
                self._doc_record(run_id, source, i + 1, base_chunks[i])
                for i in range(n_lines)
            ]

        slot_start = 0 if self.allow_before_first else 1
        slot_end = n_lines
        n_slots = slot_end - slot_start + 1

        if n_needles > n_slots:
            raise ValueError(
                f"Not enough insertion slots ({n_slots}) for {n_needles} needles."
            )

        chosen_slots = sorted(rng.sample(range(slot_start, slot_end + 1), k=n_needles))

        shuffled_needles = needles[:]
        rng.shuffle(shuffled_needles)

        slot_to_needle = {slot: shuffled_needles[i] for i, slot in enumerate(chosen_slots)}

        out: List[Dict[str, Any]] = []
        cur_id = 1

        for i in range(n_lines + 1):
            if i in slot_to_needle:
                n = slot_to_needle[i]
                out.append(self._needle_record(
                    run_id=run_id,
                    source_file=source,
                    chunk_id=cur_id,
                    text=n["needle_text"],
                    needle_item_id=n["needle_item_id"],
                    insertion_slot=i,
                ))
                cur_id += 1

            if i < n_lines:
                out.append(self._doc_record(run_id, source, cur_id, base_chunks[i]))
                cur_id += 1

        return out

    def _doc_record(self, run_id: str, source_file: str, chunk_id: int, text: str) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "source_file": source_file,
            "evidence": chunk_id,
            "chunk_type": "document",
            "text": text,
            "needle_item_id": None,
            "insertion_slot": None,
        }

    def _needle_record(
        self,
        run_id: str,
        source_file: str,
        chunk_id: int,
        text: str,
        needle_item_id: str,
        insertion_slot: int,
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "source_file": source_file,
            "evidence": chunk_id,
            "chunk_type": "needle",
            "text": text,
            "needle_item_id": needle_item_id,
            "insertion_slot": insertion_slot,
        }

    # def _get_source_name(self, source_file: str) -> str:
    #     """Extract source name as X_rand_book_Y from path like rand_shuffle_X/rand_book_Y.txt"""
    #     parts = source_file.split(os.sep)
        
    #     # Fallback if path doesn't have expected structure
    #     if len(parts) < 2:
    #         return os.path.splitext(parts[-1])[0]
        
    #     folder_name = parts[-2]  # e.g., "rand_shuffle_10000"
    #     print(folder_name)
    #     file_name = os.path.splitext(parts[-1])[0]  # e.g., "rand_book_1"
        
    #     # Extract the number from rand_shuffle_X
    #     if not folder_name.startswith("rand_shuffle_"):
    #         return file_name
        
    #     folder_number = folder_name.replace("rand_shuffle_", "")
        
    #     return f"{folder_number}_{file_name}"
    def _get_source_name(self, source_file: str) -> str:
        """Extract source name as X_rand_book_Y from path like rand_shuffle_X/rand_book_Y.txt"""
        path = Path(source_file)
        folder_name = path.parent.name
        file_name = path.stem
        
        folder_number = folder_name.replace("rand_shuffle_", "")
        return f"{folder_number}_{file_name}"