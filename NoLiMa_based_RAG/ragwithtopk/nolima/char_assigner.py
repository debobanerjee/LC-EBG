from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class CharAssigner:
    """
    Assigns a stable {CHAR} per base_id, in the order base_ids are first encountered.

    Rule matches your current code:
    - base_id = id with trailing "Inv" removed (if present)
    - when a new base_id appears, pick character_set[k % len(character_set)]
      where k = number of base_ids assigned so far
    - id and idInv share the same base_id => same CHAR
    """
    assigned: Dict[str, str] = field(default_factory=dict)

    def base_id(self, task_id: str) -> str:
        return task_id[:-3] if task_id.endswith("Inv") else task_id

    def get_char(self, task_id: str, character_set: List[str]) -> str:
        base = self.base_id(task_id)
        if base not in self.assigned:
            if character_set:
                k = len(self.assigned)
                self.assigned[base] = character_set[k % len(character_set)]
            else:
                self.assigned[base] = ""
        return self.assigned[base]
