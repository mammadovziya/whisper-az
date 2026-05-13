# Azerbaijani Whisper ASR — Benchmark & LoRA Fine-Tune

A reproducible benchmark of OpenAI Whisper (`tiny` → `large-v3`) plus Meta MMS-1B on Azerbaijani speech, and a LoRA-fine-tuned `whisper-small-az` adapter trained on Common Voice.

## Why

Azerbaijani (`az`, ~10M speakers) is a low-resource Turkic language. Whisper claims `az` support since v2, but no public benchmark documents how well the various sizes actually transcribe Azerbaijani, where they fail, or whether modest fine-tuning helps. This repo fills that gap.

## Status

Work in progress — see [`paper.md`](paper.md) for the running writeup and [`results/benchmark.json`](results/benchmark.json) for the latest numbers.

## Install

This project targets Python 3.10+ on a CUDA-capable GPU (developed on RTX 5070 Laptop, 8 GB VRAM, Blackwell / CUDA 12.6+).

**1. PyTorch with the right CUDA build** — install separately from PyPI's CUDA index because the right wheel depends on your driver. For Blackwell (RTX 50-series, compute capability `sm_120`):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ Blackwell needs **cu128 or cu130** wheels, not cu126. The cu126 build only includes kernels up to sm_90 (Hopper) and will warn `sm_120 is not compatible` on RTX 50-series.

For older NVIDIA cards, swap `cu128` for `cu121` or whatever matches your driver.

**2. The rest**

```powershell
pip install -e .
```

## Layout

```
src/
  data.py             # CV-17 (az) and FLEURS (az_az) loaders + shared text normalizer
  benchmark.py        # Whisper tiny..medium via HF + large-v3 via faster-whisper
  eval_mms.py         # MMS-1B baseline
  train_lora.py       # PEFT/LoRA fine-tune on CV-az
  analyze_errors.py   # error dump + manual taxonomy
  push_to_hub.py      # HF upload helper
results/
  benchmark.json      # canonical WER/CER table
  error_analysis.csv  # categorized FLEURS-az errors
models/
  whisper-small-az-lora/   # fine-tuned LoRA adapter (gitignored; on HF Hub)
```

## Reproduce

```powershell
# Smoke test: ~5 min
python -m src.benchmark --model whisper-tiny --dataset fleurs --max-samples 20

# Full benchmark: ~3-6 h on RTX 5070
python -m src.benchmark --model all --dataset all

# MMS baseline
python -m src.eval_mms --dataset all

# LoRA fine-tune (~6-10 h on RTX 5070)
python -m src.train_lora

# Re-evaluate fine-tuned model
python -m src.benchmark --model whisper-small-az-lora --dataset all
```

## Results

Will be filled in as runs complete. See [`results/benchmark.json`](results/benchmark.json) for the source-of-truth numbers.

## License

Apache 2.0 for code. Datasets retain their original licenses (Common Voice CC0, FLEURS CC-BY-4.0).
