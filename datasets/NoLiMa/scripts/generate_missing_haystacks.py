"""
Generate missing haystack sizes: 150k, 250k, 350k, 450k, 550k, 750k, 850k, 950k.

Uses the same algorithm as random_book_gen_char_v2.ipynb:
  - Randomly picks books from III-filter
  - Picks a random starting line and accumulates chunks of >= 1000 chars
  - Truncates to exactly TOTAL_CHARS characters
  - Creates 5 rand_book_X.txt files per size

Uses a different seed (43) to avoid entangling with the original generation's
random state (which used seed 42 and processed all sizes sequentially).
"""

import random
import os

# Source books from III-filter
_DATASET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK_DIR = os.path.join(_DATASET_DIR, "haystacks", "books", "III-filter")
OUTPUT_DIR = os.path.join(_DATASET_DIR, "haystacks")

list_of_books = os.listdir(BOOK_DIR)
# Filter to only .txt files (exclude .DS_Store etc.)
list_of_books = [b for b in list_of_books if b.endswith(".txt")]
list_of_books.sort()  # Deterministic ordering

print(f"Source books: {list_of_books}")

RAND_BOOKS = 5
MIN_LEN_CHAR = 1000  # minimum chunk size per pick
random.seed(43)       # Different seed from original (42) to keep independence

# The missing sizes
# char_range_list = [150_000, 250_000, 350_000, 450_000, 550_000, 750_000, 850_000, 950_000]
char_range_list = [650_000]

for TOTAL_CHARS in char_range_list:
    for j in range(RAND_BOOKS):
        current_chars = 0
        new_book = ""

        while current_chars < TOTAL_CHARS:
            book = random.choice(list_of_books)
            adding_part = ""

            with open(os.path.join(BOOK_DIR, book), "r") as f:
                lines = f.readlines()
                line_no = random.randint(1, len(lines) - 1)

                while len(adding_part) + 1 < MIN_LEN_CHAR and line_no < len(lines):
                    if (lines[line_no].startswith("Chapter") or
                        lines[line_no].startswith("CHAPTER") or
                        lines[line_no].startswith("...") or
                        lines[line_no].endswith("...\n") or
                        lines[line_no].strip() == ""):
                        line_no += 1
                        continue

                    adding_part += lines[line_no]
                    line_no += 1

            # Add the chunk and update char count
            new_book += adding_part
            current_chars += len(adding_part)

        out_dir = os.path.join(OUTPUT_DIR, f"rand_shuffle_{TOTAL_CHARS}")
        os.makedirs(out_dir, exist_ok=True)
        content = new_book[:TOTAL_CHARS]
        with open(os.path.join(out_dir, f"rand_book_{j+1}.txt"), "w") as f:
            f.write(content)
        print(f"rand_shuffle_{TOTAL_CHARS}/rand_book_{j+1}.txt -> {len(content)} chars written")

print("\nDone! All missing haystack sizes generated.")
