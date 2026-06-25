# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Any, Dict, List, Literal, Optional
# from ragwithtopk.utils.io import load_json, write_json
# from ragwithtopk.nolima.char_assigner import CharAssigner

# QType = Literal["all", "onehop", "twohop"]
# TestPick = Literal["first", "sorted_first"]

# @dataclass
# class QuestionExtractor:
#     needle_set_path: str

#     def extract(
#         self,
#         reasoning_type: str,
#         qtype: QType = "all",
#         test_pick: TestPick = "first",
#         out_path: Optional[str] = None,
#     ) -> List[Dict[str, Any]]:
#         """
#         Matches your current behavior:
#         - filters by reasoning_type
#         - skips inverse examples (id ending with "Inv")
#         - chooses exactly ONE tests[...] entry:
#             - "first": next(iter(tests)) (JSON insertion order)
#             - "sorted_first": sorted(tests.keys())[0]
#         - replaces {1}, {2}, ... with chosen test's input_args
#         - replaces {CHAR} using same CharAssigner logic (aligned with needles)
#         """
#         data = load_json(self.needle_set_path)
#         data = [item for item in data if item.get("reasoning_type") == reasoning_type]

#         assigner = CharAssigner()
#         out: List[Dict[str, Any]] = []

#         for item in data:
#             task_id = item.get("id", "")
#             if task_id.endswith("Inv"):
#                 continue  # important requirement

#             character_set = item.get("character_set", []) or []
#             questions = item.get("questions", {}) or {}
#             tests = item.get("tests", {}) or {}
#             if not tests:
#                 continue

#             if test_pick == "first":
#                 chosen_test_key = next(iter(tests))
#             elif test_pick == "sorted_first":
#                 chosen_test_key = sorted(tests.keys())[0]
#             else:
#                 raise ValueError("test_pick must be 'first' or 'sorted_first'")

#             input_args = tests[chosen_test_key].get("input_args", []) or []
#             chosen_char = assigner.get_char(task_id, character_set)

#             for q_key, q_tmpl in questions.items():
#                 q = q_tmpl

#                 # Replace {1}, {2}, ... based on input_args
#                 for idx, arg in enumerate(input_args, start=1):
#                     q = q.replace(f"{{{idx}}}", str(arg))

#                 # Replace {CHAR}
#                 q = q.replace("{CHAR}", chosen_char)

#                 out.append({
#                     "task_id": task_id,
#                     "question_key": q_key,
#                     "test_key": chosen_test_key,
#                     "question": q
#                 })

#         if qtype == "onehop":
#             out = [q for q in out if "onehop" in q["question_key"]]
#         elif qtype == "twohop":
#             out = [q for q in out if "twohop" in q["question_key"]]

#         if out_path:
#             write_json(out_path, out)

#         return out

# qe = QuestionExtractor("datasets/NoLiMa/needlesets/needle_set.json")
# out = qe.extract(reasoning_type="commonsense_knowledge", qtype="all", test_pick="first")
# print(out)  # print first 2 for sanity check

# ...existing code...
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import random

from ragwithtopk.utils.io import load_json, write_json

QType = Literal["all", "onehop", "twohop"]
TestPick = Literal["first", "random"]

@dataclass
class QuestionExtractor:
    needle_set_path: str

    def extract(
        self,
        reasoning_type: str,
        qtype: QType = "all",
        test_pick: TestPick = "first",
        out_path: Optional[str] = None,
        *,
        only_first_test: bool = True,
        include_q_id: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Build questions by instantiating templates with input_args from tests.

        Key behavior (as requested):
        - Can filter by reasoning_type (e.g., commonsense_knowledge)
        - Can filter to onehop/twohop
        - Uses ONLY the first test object (deterministic) when only_first_test=True
        - Adds q_id sequentially when include_q_id=True
        """
        data: List[Dict[str, Any]] = load_json(self.needle_set_path)
        out: List[Dict[str, Any]] = []

        rng = random.Random(0)

        for item in data:
            if item.get("reasoning_type") != reasoning_type:
                continue

            task_id = item.get("id")
            questions: Dict[str, str] = item.get("questions", {})
            tests: Dict[str, Dict[str, Any]] = item.get("tests", {}) or {}
            char_set: List[str] = item.get("character_set", []) or []

            if not questions or not tests or not char_set:
                continue

            # Decide test (your request: first object)
            if only_first_test:
                chosen_test_key = next(iter(tests.keys()))
            else:
                if test_pick == "first":
                    chosen_test_key = next(iter(tests.keys()))
                else:
                    chosen_test_key = rng.choice(list(tests.keys()))

            input_args = tests[chosen_test_key].get("input_args", []) or []
            chosen_char = char_set[0]  # keep deterministic; adjust if you want random

            for q_key, q_tmpl in questions.items():
                # filter to onehop/twohop
                if qtype != "all" and q_key != qtype:
                    continue
                if q_key not in ("onehop", "twohop"):
                    continue

                q = q_tmpl

                # Replace {1}, {2}, ... using input_args from the chosen test
                for idx, arg in enumerate(input_args, start=1):
                    q = q.replace(f"{{{idx}}}", str(arg))

                # Replace {CHAR}
                q = q.replace("{CHAR}", chosen_char)

                rec = {
                    "task_id": task_id,
                    "question_key": q_key,
                    "test_key": chosen_test_key,
                    "question": q,
                    "relevant_needle_id": task_id,
                }
                if include_q_id:
                    rec["q_id"] = f"{task_id}_{q_key}"

                out.append(rec)

        if out_path:
            write_json(out_path, out)

        return out


# qe = QuestionExtractor("datasets/NoLiMa/needlesets/needle_set.json")
# out = qe.extract(
#         reasoning_type="commonsense_knowledge",
#         qtype="all",          # will still only keep onehop/twohop due to filter
#         test_pick="first",
#         only_first_test=True,
#         include_q_id=True,
#     )
# print(out)