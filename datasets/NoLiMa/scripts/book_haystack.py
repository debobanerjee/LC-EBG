# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os, re, ast

from typing import List, Callable, Tuple, Union

import numpy as np
import hashlib
import random
import copy

# Optional RAG imports - only needed if use_rag=True
try:
    from rag_pipeline import RAGPipeline
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    RAGPipeline = None

class BookHaystack:
    def __init__(
        self,
        book_path: str
    ) -> None:
        self.book_path = book_path

        if not os.path.exists(book_path):
            raise FileNotFoundError(f"Book path {book_path} does not exist")
        
        if book_path.endswith('.txt'):
            with open(book_path, 'r', encoding="utf-8") as f:
                self.text = f.read()
        else:
            raise ValueError(f"Book path {book_path} is not supported")
        
        self.text_encoded = None


    def get_hash(self):
        return hashlib.sha256(self.text.encode()).hexdigest()
    
    
    def _split_haystack(self, depth, shift):
        context_length = len(self.text)
        start_pos = 0
        end_pos = min(context_length, len(self.text))

        # Compute target insert position
        insert_pos = start_pos + int(np.round(context_length * depth)) + shift
        insert_pos = max(start_pos, min(end_pos, insert_pos))

        # Find line boundaries
        line_start = self.text.rfind("\n", start_pos, insert_pos) + 1
        line_end = self.text.find("\n", insert_pos, end_pos)
        if line_end == -1:
            line_end = end_pos

        line = self.text[line_start:line_end]

        # Regex to detect line number prefix
        match = re.match(r"^(\d+:\s*)", line)
        if match:
            prefix_len = len(match.group(1))
            min_insert_pos = line_start + prefix_len
        else:
            min_insert_pos = line_start

        # Ensure insert_pos is inside line, but not before prefix
        insert_pos = max(min_insert_pos, min(insert_pos, line_end))

        # --- Snap to nearest word boundary ---
        # If we're inside a word, move right to the next boundary
        while insert_pos < line_end and re.match(r"[A-Za-z0-9]", self.text[insert_pos]):
            insert_pos += 1
        # If that pushed us to end of line, move left instead
        while insert_pos > min_insert_pos and re.match(r"[A-Za-z0-9]", self.text[insert_pos-1]):
            insert_pos -= 1

        # Split and insert
        pre_haystack = self.text[start_pos:insert_pos]
        post_haystack = self.text[insert_pos:end_pos]

        return pre_haystack, post_haystack, insert_pos
    
    def _generate_w_needle_placement(
        self, 
        needle: str, 
        # token_count_func: Callable,
        # encoding_func: Callable,
        # decoding_func: Callable,
        # context_length: int,
        shift: int = 0,
        depth: float = 0.5,
        static_depth: float = -1,
        # use_line_numbers_in_haystack: bool = False,
        insert_at_independent_lines: bool = False,
        use_char_count: bool = False,
        use_rag: bool = False,
        rag = None,  # RAGPipeline = None (optional, avoid type hint)
        rag_character_budget: int = 500_000,
        query: str = None
    ) -> dict:
        
        # Check RAG availability if requested
        if use_rag and not RAG_AVAILABLE:
            raise ImportError("RAG pipeline not available. Install required dependencies for RAG support.")

        if use_char_count:
            if insert_at_independent_lines:
                needle_position = random.choice(range(int(len(self.text.split("\n")) * depth))) + shift
                text = copy.deepcopy(self.text).split("\n")
                text.insert(needle_position, needle)
            else:
                if static_depth == -1:
                    # Split the haystack around insert_pos 
                    pre_haystack, post_haystack, insert_pos = self._split_haystack(depth, shift)
    
                    text = pre_haystack + " " + needle + " " + post_haystack
                    needle_position = len((pre_haystack + " " + needle).split("\n"))-1
    
                
                
            context_length_wo_needle = sum(i+1 for i in range(len(self.text.split("\n"))))
            context_length_w_needle = sum(i+1 for i in range(len(text.split("\n"))))

            text = "\n".join(f"{i}: {line}" for i, line in enumerate(text))

            if use_rag:
                # rag.add_document(text)
                rag_ret_chunks, rag_ret_scores, rag_ret_char_count = rag.retrieve_upto_char_budget(query, char_budget=rag_character_budget)
                # Insert needle among the retrieved top_k lines
                rag_haystack_w_needle, rag_needle_positions = rag.get_haystack_w_needle(rag_ret_chunks, [needle])
                
                context_length_w_needle = sum(i+1 for i in range(len(rag_haystack_w_needle)))
                text = "\n".join(f"{i}: {line.text}" for i, line in enumerate(rag_haystack_w_needle))
                    
                return {
                    "text": text,
                    # "rag_text": rag_text,
                    "needle_line_num": rag_needle_positions[0],
                    # "rag_needle_line_num": rag_needle_line_num,
                    # "static_depth": static_depth,
                    # "insert_pos": insert_pos,
                    "depth": depth,
                    "context_length_wo_needle": rag_ret_char_count,
                    "context_length_w_needle": context_length_w_needle,

                }
            # if use_line_numbers_in_haystack:
                # numbered_lines = [f"{i}: {line}" for i, line in enumerate(text.split("\n"))]
                # text = '\n'.join(numbered_lines)
            # print(f"{len(self.text)} {len(text)}")
            return {
                "text": text,
                "needle_line_nums": needle_position,
                # "static_depth": static_depth,
                # "insert_pos": insert_pos,
                "depth": depth,
                "context_length_wo_needle": context_length_wo_needle,
                "context_length_w_needle": context_length_w_needle
            }
        
        else:
            raise NotImplementedError

    def generate_w_needle_placement(
        self, 
        needle: str, 
        token_count_func: Callable,
        encoding_func: Callable,
        decoding_func: Callable,
        context_length: int,
        shift: int = 0,
        depth: float = 0.5,
        static_depth: float = -1,
        distractor: Union[str, None] = None,
        distractor_free_zone: float = 0.2,
        use_line_numbers_in_haystack: bool = False,
        use_char_count: bool = False,
        use_rag: bool = False,
        rag = None,  # RAGPipeline = None (optional, avoid type hint)
        query: str = None
    ) -> dict:
        
        # Check RAG availability if requested
        if use_rag and not RAG_AVAILABLE:
            raise ImportError("RAG pipeline not available. Install required dependencies for RAG support.")
        if distractor is None:
            return self._generate_w_needle_placement(
                needle, 
                token_count_func,
                encoding_func,
                decoding_func,
                context_length,
                shift,
                depth,
                static_depth, 
                use_line_numbers_in_haystack,
                use_char_count,
                use_rag,
                rag,
                query
            )
        elif static_depth == -1:
            if distractor_free_zone < 0 or distractor_free_zone > 0.25:
                raise ValueError("Distractor free zone should be between 0 and 0.25")
            
            left_available_span = max(0, depth - 2 * distractor_free_zone)
            right_available_span = max(0, 1 - (depth + 2 * distractor_free_zone))

            dist_depth = np.random.uniform(0, left_available_span + right_available_span)
            if dist_depth > left_available_span:
                dist_depth = depth + distractor_free_zone + (dist_depth - left_available_span)
            else:
                dist_depth += distractor_free_zone
            
            if context_length < 500:
                if depth < 0.6:
                    dist_depth = 0.7
                elif depth > 0.75:
                    dist_depth = 0.6
                else:
                    dist_depth = 0.5

            placement_w_distractor = self._generate_w_needle_placement(
                distractor, 
                token_count_func,
                encoding_func,
                decoding_func,
                context_length,
                shift,
                dist_depth,
                static_depth,
                use_line_numbers_in_haystack,
                use_char_count,
                use_rag,
                rag,
                query
            )

            distr_char_pos = placement_w_distractor["text"].index(distractor)
            distr_placement_findspan = (max(distr_char_pos-50, 0), distr_char_pos-1)
            if context_length < 500:
                distr_placement_findspan = (max(distr_char_pos-25, 0), distr_char_pos-1)
            

            placement_w_needle = self._generate_w_needle_placement(
                needle, 
                token_count_func,
                encoding_func,
                decoding_func,
                context_length,
                shift,
                depth,
                static_depth,
                use_line_numbers_in_haystack,
                use_char_count,
                use_rag,
                rag,
                query
            )

            if placement_w_needle["text"].count(placement_w_distractor["text"][distr_placement_findspan[0]:distr_placement_findspan[1]]) > 1:
                raise IndexError("Multiple spans found with same pre-distractor text")

            distractor_loc = placement_w_needle["text"].index(placement_w_distractor["text"][distr_placement_findspan[0]:distr_placement_findspan[1]]) + (49 if context_length >= 500 else 23)
            placement_w_needle["text"] = placement_w_needle["text"][:distractor_loc] + " " + distractor + "\n" + placement_w_needle["text"][distractor_loc:]
            placement_w_needle["distractor_depth"] = dist_depth
            return placement_w_needle
        else:
            ValueError("Static depth is not supported with distractor")