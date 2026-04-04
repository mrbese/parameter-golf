# BESE Tokenizer + Extended Value Embeddings

**val_bpb: TBD** (single seed) | **~TBD MB** | 8xH100 SXM, 600s

This submission introduces two novel components to the PR #1019 architecture:

1. **BESE Tokenizer** (287 vocab) — a structured tokenizer built from first principles, replacing SentencePiece-1024
2. **Extended Value Embeddings** — VE on layers 4-10 (7 layers) instead of the standard 9-10 (2 layers)

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 1337 | TBD | TBD | TBD | **TBD** | TBD |

---

## Novel Contribution 1: BESE Tokenizer

BESE (Byte-Efficient Structured Encoding) replaces the standard SentencePiece-1024 tokenizer with a 287-token vocabulary built from a structured 38-token base alphabet:

- **8 single-letter tokens**: `e t a o i n s r` (the 8 most frequent English letters)
- **5 T9-style consonant groups**: remaining consonants grouped by phone keypad layout
- **4 positional tokens**: encode character position within groups
- **Special tokens**: space, newline, digit prefix, uppercase prefix, BOS/EOS/PAD/UNK
- **10 digit tokens**: 0-9
- **249 BPE merges**: learned on FineWeb data, fully absorbing group/position tokens

### Tokenization Efficiency

| Tokenizer | Vocab | Tokens/Byte | Embedding Params |
|-----------|-------|-------------|-----------------|
| SentencePiece-1024 | 1,024 | 0.76 | 524,288 |
| **BESE-287** | **287** | **0.51** | **146,944** |

BESE achieves **33% fewer tokens per byte**, meaning the model sees more content per context window. The embedding table is 72% smaller, freeing parameter budget for the transformer layers.

### Design Philosophy

The alphabet was designed independently from first principles, drawing on:
- **T9 phone keyboards**: grouping consonants by frequency and phonetic similarity
- **Huffman coding intuition**: most frequent letters get dedicated tokens
- **Bionic Reading**: the idea that partial information (first letters) is sufficient for pattern recognition

The tokenizer is fully self-contained — no external SentencePiece dependency needed.

## Novel Contribution 2: Extended Value Embeddings (VE)

Standard implementation applies Value Embeddings only to the final 2 layers (9-10). We extend VE to **layers 4-10** (7 layers), re-injecting token identity deeper into the network.

**Motivation**: As tokens pass through attention layers, their individual identity gets mixed away. By re-injecting token identity at layer 4 (before the model has fully abstracted away token-level features), the model maintains a stronger signal of what each token actually is throughout the computation.

**Cost**: ~200KB extra at INT6 quantization, well within the 16MB budget.

| Config | VE Layers | VE Parameters |
|--------|-----------|--------------|
| Standard (PR #1019) | 9, 10 | ~204K |
| **This submission** | **4, 5, 6, 7, 8, 9, 10** | **~714K** |

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Tokenizer | BESE-287 (38 base + 249 BPE) | **Novel** |
| Value Embeddings | Layers 4-10, dim=128 | **Novel extension** |
| Layers | 11 | PR #1019 |
| XSA | All 11 layers | PR #1019 |
| Quantization | GPTQ int6 + LZMA preset=9 | PR #1019 |
| Optimizer | Muon (WD=0.04) | PR #1019 |
| BigramHash | 3072 x 112 | PR #1019 |
| EMA/SWA | decay=0.997 / every 50 steps | PR #1019 |
| TARGET_MB | 15.9 (selective pruning) | PR #1019 |

## The Story Behind BESE

This tokenizer was built by a founder/engineer with no formal ML research background, arriving at established concepts independently through everyday analogies:

- **T9 phone keyboards** inspired the consonant grouping — pressing one key to represent multiple related letters
- **QWERTY keyboard layout** informed the frequency-based letter selection — why do the most-used letters get prime real estate?
- **Bionic Reading** (the speed-reading technique that bolds first letters) suggested that partial character information could be enough for pattern recognition
- **Huffman coding** principles emerged naturally from thinking about efficiency — give the most common symbols the shortest codes

The result was a tokenizer design that, when validated against the literature, independently rediscovered mutual information minimization and hierarchical encoding — but arrived there from a completely different starting point than traditional NLP research.

## Data Preparation

BESE requires pre-tokenized training data. The preparation pipeline:
1. Decode original SentencePiece-1024 binary shards back to text
2. Train BPE merges on 50K documents from the training set
3. Re-encode all shards with the BESE tokenizer
4. Write new binary shards in the same format

Total prep time: ~50 minutes on a 32-vCPU CPU pod (one-time cost, not counted in the 10-minute training budget).
