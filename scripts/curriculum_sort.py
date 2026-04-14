#!/usr/bin/env python3
"""Sort training docs easy→hard for curriculum learning.

Difficulty is a composite of average word length, vocabulary richness,
and average sentence length. Easy documents are placed first so the model
sees simple patterns early in training.
"""
import json
import argparse
from pathlib import Path


def difficulty_score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    vocab_richness = len(set(words)) / len(words)
    sentences = max(text.count('.') + text.count('!') + text.count('?'), 1)
    avg_sentence_len = len(words) / sentences
    return (avg_word_len / 10) * 0.3 + vocab_richness * 0.4 + (avg_sentence_len / 30) * 0.3


def main():
    ap = argparse.ArgumentParser(description="Sort JSONL docs by difficulty (easy first)")
    ap.add_argument('--input', required=True, help="Input JSONL file with 'text' field")
    ap.add_argument('--output', required=True, help="Output sorted JSONL file")
    args = ap.parse_args()

    docs = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append((difficulty_score(obj['text']), obj))

    docs.sort(key=lambda x: x[0])

    with open(args.output, 'w') as f:
        for _, obj in docs:
            f.write(json.dumps(obj) + '\n')

    print(f"Sorted {len(docs)} docs by difficulty: easy first")


if __name__ == '__main__':
    main()
