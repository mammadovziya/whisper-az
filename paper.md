# Whisper on Azerbaijani: A Benchmark and a LoRA Fine-Tune

*Draft. Numbers will be filled in as runs complete.*

## Abstract

TBD.

## 1. Introduction

Azerbaijani (`az`) is a low-resource Turkic language with ~10M native speakers. OpenAI's Whisper has nominally supported `az` since v2, but no rigorous public benchmark documents per-size performance, failure modes, or whether modest fine-tuning improves it. This work contributes (i) a reproducible benchmark of Whisper `{tiny, base, small, medium, large-v3}` and Meta MMS-1B on FLEURS-az and Common Voice 17 az, (ii) an error taxonomy from manual categorization of ~200 FLEURS-az errors, and (iii) a LoRA-fine-tuned `whisper-small-az` adapter.

## 2. Setup

### 2.1 Datasets

- **FLEURS-az** (`google/fleurs`, config `az_az`). Test split.
- **Common Voice 17 az** (`mozilla-foundation/common_voice_17_0`, config `az`). Test split for evaluation; train/validation for LoRA.

### 2.2 Models evaluated

| Family | Sizes | Backend |
|---|---|---|
| Whisper | tiny, base, small, medium | `transformers` fp16 |
| Whisper | large-v3 | `faster-whisper` int8_float16 |
| MMS | 1B-all (`target_lang=aze`) | `transformers` fp16 |

### 2.3 Normalization

All hypothesis and reference strings are passed through `transformers.models.whisper.tokenization_whisper.BasicTextNormalizer` before scoring. WER/CER computed with `evaluate` (jiwer backend).

### 2.4 Hardware

Single RTX 5070 Laptop GPU, 8 GB VRAM, Windows 11. fp16 inference for `tiny`–`medium`; int8 quantization for `large-v3`. LoRA-only fine-tuning.

## 3. Benchmark Results

TBD — see `results/benchmark.json`.

## 4. Error Taxonomy

TBD — see `results/error_analysis.csv`.

Categories considered: numerals, named entities, code-switching (Russian/Turkish/English), Azerbaijani-specific letters (`ə ğ ı ö ş ü ç`), dialect/register, audio quality, other.

## 5. LoRA Fine-Tune

### 5.1 Setup

PEFT LoRA, `r=32`, `α=64`, dropout `0.05`. Targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`. Base: `openai/whisper-small`. Train: CV-17 az `train`. Eval: CV-17 az `validation`. 5 epochs, lr `1e-4`, warmup 100 steps, fp16, gradient checkpointing, batch 8 × accum 2.

### 5.2 Results

TBD.

## 6. Discussion

TBD.

## 7. Limitations

- Common Voice transcripts are crowd-sourced and contain noise.
- `large-v3` evaluated under int8 quantization (~1–2 WER points worse than fp16 in published evals); disclosed in tables.
- Single-GPU laptop setup; no multi-seed runs.
- LoRA target modules and rank chosen by convention; not swept.

## 8. Release

- Model: `huggingface.co/<user>/whisper-small-az-lora`
- Code: GitHub link TBD.
