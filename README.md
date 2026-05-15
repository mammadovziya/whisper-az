# whisper-az

<p align="center">
  Azerbaijani ASR benchmark for Whisper and MMS-1B
</p>

<p align="center">
  Benchmarking OpenAI Whisper (`tiny` → `large-v3`) on Azerbaijani speech + LoRA fine-tuning experiments.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue">
  <img src="https://img.shields.io/badge/CUDA-12.8-green">
  <img src="https://img.shields.io/badge/license-Apache_2.0-orange">
  <img src="https://img.shields.io/badge/status-research-informational">
</p>

---

## The Problem

OpenAI Whisper officially supports Azerbaijani (`az`), but there is almost no public evaluation showing:

- how well different Whisper sizes actually work
- where smaller models fail
- how strong multilingual interference is
- whether lightweight fine-tuning improves performance

This repository benchmarks Whisper (`tiny` → `large-v3`) and Meta MMS-1B on Azerbaijani speech and provides a LoRA fine-tuned Whisper adapter trained on Common Voice.

---

## Current Findings

Initial experiments show that Azerbaijani performance scales dramatically with Whisper model size.

Smaller multilingual models exhibit:

- severe Turkish orthographic interference
- missing Azerbaijani diacritics
- dropped suffixes
- unstable word segmentation

while `whisper-large-v3` and MMS-1B become substantially more usable.

---

## Benchmark Results (FLEURS Azerbaijani)

Evaluation on the FLEURS `az_az` test split (`n = 923`).

| Model | Params | WER ↓ | CER ↓ | Runtime |
|---|---:|---:|---:|---:|
| Whisper Tiny | 39M | 101.8% | 50.4% | 68s |
| Whisper Base | 74M | 82.1% | 30.1% | 65s |
| Whisper Small | 244M | 51.7% | 14.4% | 161s |
| Whisper Medium | 769M | 34.3% | 9.0% | 597s |
| Whisper Large-v3 | 1550M | **21.7%** | 5.9% | 1045s |
| MMS-1B (Azerbaijani) | 1B | 23.8% | **5.3%** | 91s |

### Key Observations

- Small Whisper models perform very poorly on Azerbaijani.
- Accuracy improves almost monotonically with model scale.
- `whisper-large-v3` achieves the best WER.
- MMS-1B achieves slightly better CER despite worse WER.
- Character-level accuracy improves much faster than word-level accuracy.

Source-of-truth results are stored in:

```text
results/benchmark.json
````

---

## Benchmark Results (Common Voice 25 Azerbaijani)

Evaluation on the Common Voice 25.0 `az` test split (`n = 126`), loaded from the Mozilla Data Collective tarball.

| Model | Params | WER ↓ | CER ↓ | Runtime |
|---|---:|---:|---:|---:|
| Whisper Tiny | 39M | 101.7% | 62.2% | 11s |
| Whisper Base | 74M | 102.0% | 44.6% | 12s |
| Whisper Small | 244M | 62.8% | 20.0% | 16s |
| Whisper Medium | 769M | 42.1% | 12.2% | 41s |
| Whisper Large-v3 | 1550M | **26.5%** | 6.9% | 75s |
| MMS-1B (Azerbaijani) | 1B | 28.3% | **6.3%** | 15s |

### Key Observations

- `whisper-base` is worse than `whisper-tiny` on CV-25 (hallucinates long outputs, pushing WER past 100%).
- The first useful jump in model scale is `base → small`.
- The "MMS wins on CER, Whisper-large wins on WER" pattern from FLEURS holds on CV-25 too — a consistent systematic difference in failure modes across two independent test sets.

---

## Domain Shift: FLEURS → Common Voice 25

Every model is worse on CV-25 (crowd-sourced, noisy) than on FLEURS-az (clean read speech). The shift is largest for the smallest models and shrinks with scale — larger models are more robust to recording conditions and speaker variation.

| Model | FLEURS WER | CV-25 WER | Δ WER | FLEURS CER | CV-25 CER | Δ CER |
|---|---:|---:|---:|---:|---:|---:|
| Whisper Tiny | 101.8% | 101.7% | -0.1 | 50.4% | 62.2% | +11.8 |
| Whisper Base | 82.1% | 102.0% | **+19.9** | 30.1% | 44.6% | +14.5 |
| Whisper Small | 51.7% | 62.8% | +11.1 | 14.4% | 20.0% | +5.6 |
| Whisper Medium | 34.3% | 42.1% | +7.8 | 9.0% | 12.2% | +3.2 |
| Whisper Large-v3 | 21.7% | 26.5% | +4.8 | 5.9% | 6.9% | +1.0 |
| MMS-1B | 23.8% | 28.3% | +4.5 | 5.3% | 6.3% | +1.0 |

---

## LoRA Fine-Tuning Results

A parameter-efficient LoRA adapter was trained on top of `openai/whisper-small`, using FLEURS-az `train` (2665 utts) + Common Voice 25 az `train` (215 utts) — ~2880 samples combined. **Only ~13M trainable parameters (5.1% of the base model), 51 MB on disk.** Training took ~58 min on an RTX 5070 Laptop (5 epochs, batch 8 × accum 2, lr 1e-4, FLEURS validation eval-loss 1.40 → 0.51).

| Model | Params (trainable) | FLEURS WER | FLEURS CER | CV-25 WER | CV-25 CER |
|---|---:|---:|---:|---:|---:|
| Whisper Small (base) | 244M | 51.7% | 14.4% | 62.8% | 20.0% |
| **Whisper Small + LoRA (ours)** | 244M + 13M | **35.0%** | **9.7%** | **40.7%** | **12.2%** |
| Whisper Medium (3× bigger) | 769M | 34.3% | 9.0% | 42.1% | 12.2% |
| Whisper Large-v3 (6× bigger) | 1550M | 21.7% | 5.9% | 26.5% | 6.9% |

### Key Result

The LoRA adapter brings `whisper-small` to **medium-size performance** on FLEURS (35.0% vs 34.3% WER) and **beats whisper-medium on the noisy CV-25 test set** (40.7% vs 42.1% WER) — using a base model 1/3 the size and an adapter file of just 51 MB. Relative WER reduction over baseline:

- FLEURS-az: **−32%** WER (51.7 → 35.0)
- CV-25 az:  **−35%** WER (62.8 → 40.7)

The adapter does not yet match `large-v3` or MMS-1B, but for offline deployment where compute/VRAM matters, `whisper-small + LoRA` is the strongest small-footprint option.

### Reproducing

```bash
# Sanity check (~5 min)
python -m src.train_lora --max-steps 50 --eval-steps 25 --max-train-samples 200

# Full run (~1 h on RTX 5070)
python -m src.train_lora

# Evaluate the resulting adapter on both test sets
python -m src.benchmark --model whisper-small-az-lora --lora-path models/whisper-small-az-lora --dataset fleurs
python -m src.benchmark --model whisper-small-az-lora --lora-path models/whisper-small-az-lora --dataset common_voice
```

Per-step training loss curve is in `results/train_lora_history.txt`.

---

## Example Transcription

Ground truth:

```text
Mən sabah universitetə gedəcəyəm.
```

Whisper Tiny:

```text
Ben sabah üniversiteye gideceğim.
```

Whisper Large-v3:

```text
Mən sabah universitetə gedəcəyəm.
```

This highlights a common failure mode in smaller multilingual models:
Turkish interference replacing Azerbaijani orthography.

---

## Project Goals

* Benchmark Whisper on Azerbaijani speech
* Compare scaling behavior across model sizes
* Analyze linguistic failure modes
* Evaluate MMS-1B as a multilingual baseline
* Test parameter-efficient adaptation (LoRA)
* Provide reproducible evaluation scripts

---

## Repository Layout

```text
src/
  data.py             # CV-25 (az, local MDC tarball) and FLEURS (az_az, HF) loaders
  benchmark.py        # Whisper tiny..large-v3 benchmark
  eval_mms.py         # MMS-1B baseline evaluation
  train_lora.py       # PEFT/LoRA fine-tuning
  analyze_errors.py   # error analysis pipeline
  push_to_hub.py      # Hugging Face upload helper

results/
  benchmark.json                  # WER/CER table (model × dataset)
  error_analysis.csv              # per-sentence errors for manual taxonomy
  transcripts/                    # per-sample (ref, hyp) pairs for every run

data/
  cv25-az/                        # extracted Common Voice 25.0 MDC tarball (gitignored)
    cv-corpus-25.0-.../az/
      train.tsv, dev.tsv, test.tsv
      clips/

models/
  whisper-small-az-lora/          # fine-tuned LoRA adapter (gitignored)
```

---

## Quickstart

Clone the repository:

```bash
git clone https://github.com/yourname/whisper-az.git
cd whisper-az
```

Install dependencies:

```bash
pip install -e .
```

---

## GPU Notes

Developed and tested on:

* RTX 5070 Laptop GPU
* CUDA 12.8
* Python 3.10+

For Blackwell GPUs (RTX 50-series), install PyTorch with cu128 or newer:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> RTX 50-series GPUs require `cu128+` wheels because older builds do not include `sm_120` kernels.

---

## Reproducing Results

### Smoke Test

```bash
python -m src.benchmark --model whisper-tiny --dataset fleurs --max-samples 20
```

### Full Whisper Benchmark

```bash
python -m src.benchmark --model all --dataset all
```

### MMS-1B Baseline

```bash
python -m src.eval_mms --dataset all
```

### LoRA Fine-Tuning

```bash
python -m src.train_lora
```

### Evaluate Fine-Tuned Model

```bash
python -m src.benchmark --model whisper-small-az-lora --dataset all
```

---

## Datasets

### Common Voice 25.0 Azerbaijani

Used for:

* cross-domain benchmark evaluation (crowd-sourced speech)
* LoRA fine-tuning (combined with FLEURS-az train)

Distribution:

* As of October 2025, Mozilla distributes Common Voice exclusively through **[Mozilla Data Collective](https://mozilladatacollective.com/)** — the `mozilla-foundation/common_voice_*` HF Hub repos are no longer programmatically accessible.
* Sign up for an MDC account, request the Azerbaijani dataset, and download the tarball. Extract to `data/cv25-az/`. The loader in `src/data.py` reads directly from the extracted directory.
* CV-25 az is small: 215 train / 93 dev / 126 test utterances.

License:

* CC0

### FLEURS Azerbaijani (`az_az`)

Used for:

* primary read-speech benchmark
* cross-model comparison

License:

* CC-BY-4.0

---

## Planned Work

* Publish LoRA fine-tuned checkpoint to Hugging Face
* Expand linguistic error taxonomy (manual labeling of ~200 worst FLEURS-az errors)
* Add inference speed / VRAM benchmarks
* Evaluate distillation approaches
* Compare against SeamlessM4T and Canary

---

## License

Code:

* Apache 2.0

Datasets retain their original licenses:

* Common Voice → CC0
* FLEURS → CC-BY-4.0
