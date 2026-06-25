from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedContextFormatter:
    """
    Utilities to format retrieved chunks:
    - Sort ascending by evidence (line number)
    - Insert "..." when gaps exist
    - Return either (evidence+text) or text-only
    """

    gap_token: str = "..."

    def sort_by_evidence(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(hits, key=lambda x: int(x["evidence"]))

    def format_with_line_numbers(self, hits: List[Dict[str, Any]]) -> str:
        """
        Output:
        [Line Number: {evidence}, Text: {text}]
        ...
        [Line Number: {evidence}, Text: {text}]
        """
        hits_sorted = self.sort_by_evidence(hits)
        if not hits_sorted:
            return ""

        out_lines: List[str] = []
        prev_e: Optional[int] = None

        for h in hits_sorted:
            e = int(h["evidence"])
            t = (h.get("text") or "").strip()

            if prev_e is not None and e > prev_e + 1:
                out_lines.append(self.gap_token)

            out_lines.append(f"(Line Number: {e}, Text: {t})")
            prev_e = e

        # return "\n".join(out_lines)
        return out_lines

    def texts_only(self, hits: List[Dict[str, Any]], with_gaps: bool = True) -> List[str]:
        """
        Returns texts in ascending evidence order.
        If with_gaps=True, inserts gap_token as a separate element when gaps exist.
        """
        hits_sorted = self.sort_by_evidence(hits)
        if not hits_sorted:
            return []

        out: List[str] = []
        prev_e: Optional[int] = None

        for h in hits_sorted:
            e = int(h["evidence"])
            t = (h.get("text") or "").strip()

            if with_gaps and prev_e is not None and e > prev_e + 1:
                out.append(self.gap_token)

            out.append(t)
            prev_e = e

        return out

    def texts_only_string(self, hits: List[Dict[str, Any]], with_gaps: bool = True) -> str:
        """
        Same as texts_only(), but joined by newlines for printing/logging.
        """
        return "\n".join(self.texts_only(hits, with_gaps=with_gaps))
