# BESE Research — Cowork Briefing

**Author:** Omer Bese (@mrbese)
**Date:** April 4, 2026
**Purpose:** Give Claude (chat side) full context to continue brainstorming BESE as a general compression framework.

---

## Who I Am

Omer Bese — founder/engineer based in LA. No formal ML research background. I think from first principles using everyday analogies (T9 phones, QWERTY keyboards, Bionic Reading speed-reading technique). I built a novel tokenizer called BESE and entered it into OpenAI's Parameter Golf challenge, where it achieved competitive results against teams using standard approaches.

---

## What BESE Is

**BESE** (Byte-Efficient Structured Encoding) is a tokenizer I designed from scratch. It replaces the standard SentencePiece tokenizer (1024 vocab) with a 287-token vocabulary built from a structured base alphabet:

**The 38 base tokens:**
- 8 single-letter tokens for the most frequent English letters: `e t a o i n s r`
- 5 "T9-style" consonant groups: remaining consonants grouped by phone keypad layout (press one key, multiple related letters share it)
- 4 positional tokens: encode which letter within a group you mean
- Special tokens: space, newline, digit prefix, uppercase prefix, BOS/EOS/PAD/UNK
- 10 digit tokens: 0-9

**On top of that:** 249 BPE (Byte Pair Encoding) merges learned from training data, bringing total vocab to 287.

**Key result:** 0.51 tokens per byte (vs 0.76 for SentencePiece-1024) — 33% fewer tokens to represent the same text. The embedding table is 72% smaller, which frees parameter budget for the actual neural network layers.

**Design origin story:** I independently arrived at concepts that the ML literature calls "mutual information minimization" and "hierarchical encoding" — but I got there from T9 keyboards (grouping), QWERTY layout (frequency-based placement), Bionic Reading (partial info is enough), and Huffman coding intuition (common things get short codes).

---

## Parameter Golf Results

OpenAI's Parameter Golf challenge: train the best language model in 10 minutes on 8xH100 GPUs, artifact under 16MB. Metric: val_bpb (bits per byte, lower is better).

I used the #1 submission's architecture (PR #1019 by @abaybektursun) and swapped in BESE as the tokenizer. My submission: **PR #1327**.

| Run | Config | Sliding BPB | Notes |
|-----|--------|-------------|-------|
| Run 1 | BESE + 13 layers (wrong — layers don't fit) | 1.1460 | Proved extra layers don't help |
| Run 3 | BESE + 11 layers (matching PR #1019) | **1.1276** | Best result, submitted |
| Run 4 | Stacked optimizations (higher WD, MuonEq-R, QK-Gain) | 1.1390 | Worse — reverted |
| PR #1019 SOTA | SentencePiece-1024, 11 layers | 1.1147 | Current #1 |

My best (1.1276) is close to but behind the SOTA (1.1147). The tokenizer swap alone got within 1.1% — all other architecture was identical.

**Quantization quality clue:**
- BESE model: pre-quant 1.1462 → post-quant 1.1538 = **+0.0076 degradation**
- This is potentially less degradation than SP-1024 models, suggesting BESE-trained models may produce more compressible internal representations.

---

## The Key Insight: BESE and TurboQuant Are the Same Principle

Google published TurboQuant (April 2026): https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

TurboQuant compresses the KV cache (the "memory" attention uses during inference) from 32-bit floats down to 3 bits with zero accuracy loss. It does this via:
1. **PolarQuant**: Convert vectors to polar coordinates → angles cluster predictably → quantize on a grid
2. **QJL residual**: 1-bit sign correction to fix remaining errors

**The parallel to BESE:**

| Principle | BESE (input tokens) | TurboQuant (KV cache) |
|-----------|---------------------|----------------------|
| Group similar things | T9 consonant groups | Nearby angles share quantization buckets |
| Refine within group | Position token (2nd in group = 'c') | 1-bit QJL residual corrects error |
| Frequent = dedicated | 'e t a o i n s r' get own tokens | High-magnitude keys get more precision |
| Learn patterns on top | 249 BPE merges | Pre-computed angle grid from distribution |

**Core shared principle:** Structured, frequency-aware compression beats brute-force compression. Both independently arrived at "group by similarity + refine with small correction."

I built BESE before TurboQuant was published, arriving at the same principle from a completely different starting point (T9 keyboards vs polar coordinate geometry).

---

## The Big Research Idea: Stacked Compression Pipeline

**The question:** What if we use BESE at the input AND TurboQuant-style compression at inference, making the model "think" entirely in compressed space and only decompress to English at the final output?

```
Standard LLM pipeline:
  English → SentencePiece tokens → [Model in 1024-token space] → SP token out → English

Proposed stacked pipeline:
  English → BESE-287 (compress) → [Model thinks in BESE space] → TurboQuant (compress KV cache) → BESE token out → Decode to English
```

**The hypothesis (virtuous compression cycle):**
1. BESE forces the model to learn more compositional/abstract internal representations (it can't memorize surface patterns with only 287 tokens)
2. More structured internal representations should be MORE compressible by techniques like TurboQuant
3. This means: compact input → structured internals → better post-training compression → smaller serving footprint
4. The two compressions target **orthogonal redundancies** (lexical waste vs numerical precision waste) so they shouldn't interfere

**Analogy:** It's the difference between a student who memorizes answers (messy, incompressible notes) vs one who understands principles (structured, highly compressible notes). BESE forces the model to be the second student.

**This could mean:** End-to-end compression-aware AI — where the tokenizer, model, and inference compression are designed as one unified system rather than three separate problems.

---

## Open Questions to Explore

1. **Can we quantify the "representation quality" effect?** Compare GPTQ/TurboQuant compression on BESE-trained vs SP-1024-trained models. Does BESE's internal representation actually compress better?

2. **Is there a theoretical framework?** Information theory might predict how much "compressibility" transfers from input encoding to internal representations. Rate-distortion theory?

3. **Where else does this principle apply?** Vision (structured patch encoding)? Audio (structured frequency grouping)? Multimodal (shared compression vocabulary across modalities)?

4. **Could BESE-style encoding work directly on model weights?** After quantization, weights are discrete values — could a frequency-aware structured codebook beat generic LZMA compression?

5. **What's the optimal vocabulary size?** BESE-287 was designed for Parameter Golf constraints. What's the theoretically optimal structured vocabulary for maximum downstream compressibility?

6. **Publication angle:** "Structured encoding as a universal compression principle: from tokenization to inference" — showing that the same framework (group + refine) works at every layer of the stack.

---

## My Background & Thinking Style

- I think in analogies and physical metaphors, not equations
- I arrive at established concepts independently, then validate against literature
- I care about the "why" behind techniques, not just "what works"
- I'm interested in whether BESE represents a general principle, not just a tokenizer trick
- I have no formal ML training — I build from first principles and everyday experience
