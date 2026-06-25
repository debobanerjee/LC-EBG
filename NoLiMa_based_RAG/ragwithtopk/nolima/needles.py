from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from ragwithtopk.utils.io import load_json
from ragwithtopk.nolima.char_assigner import CharAssigner

@dataclass
class NeedleExtractor:
    needle_set_path: str

    def extract(self, reasoning_type: str) -> List[Dict[str, Any]]:
        """
        Returns a list of needle dicts:
          {
            "needle_text": str,
            "needle_item_id": str,     # original id (e.g., "0402" or "0402Inv")
            "base_id": str,
            "chosen_char": str
          }

        Matches your current behavior:
        - includes both id and idInv (no skipping)
        - id and idInv share same chosen_char
        - chosen_char is assigned in order of first-seen base_id
        """
        data = load_json(self.needle_set_path)
        data = [item for item in data if item.get("reasoning_type") == reasoning_type]

        assigner = CharAssigner()
        out: List[Dict[str, Any]] = []

        for item in data:
            task_id = item.get("id", "")
            needle_template = item.get("needle", "") or ""
            character_set = item.get("character_set", []) or []

            chosen_char = assigner.get_char(task_id, character_set)
            base_id = assigner.base_id(task_id)

            needle_text = needle_template.replace("{CHAR}", chosen_char)

            out.append({
                "needle_text": needle_text,
                "needle_item_id": task_id,
                "base_id": base_id,
                "chosen_char": chosen_char,
            })

        return out
