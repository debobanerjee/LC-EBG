# ragwithtopk/utils/rng.py
from __future__ import annotations
import hashlib

def seed_for_file(base_seed: int, file_key: str) -> int:
    """
    Stable per-file seed derived from (base_seed, file_key).
    """
    h = hashlib.sha256(f"{base_seed}:{file_key}".encode("utf-8")).hexdigest()
    # fit into 32-bit range for random.Random
    return int(h[:8], 16)
