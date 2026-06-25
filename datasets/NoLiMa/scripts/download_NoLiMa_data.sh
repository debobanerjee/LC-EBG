#!/usr/bin/env bash
# Download the upstream NoLiMa needle sets and haystacks from HuggingFace
# into this dataset directory. Run from anywhere; paths are resolved relative
# to the script's parent directory (datasets/NoLiMa/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASET_DIR/needlesets"
cd "$DATASET_DIR/needlesets"
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set.json
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_MC.json
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_ONLYDirect.json
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_hard.json
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_w_CoT.json
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_w_Distractor.json

mkdir -p "$DATASET_DIR/haystacks/rand_shuffle"
cd "$DATASET_DIR/haystacks/rand_shuffle"
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_1.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_2.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_3.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_4.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_5.txt

mkdir -p "$DATASET_DIR/haystacks/rand_shuffle_long"
cd "$DATASET_DIR/haystacks/rand_shuffle_long"
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_1.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_2.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_3.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_4.txt
wget -c https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_5.txt
