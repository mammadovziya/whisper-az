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
| Whisper Large-v3 | 1550M | **21.7%** | 5.9% | — |
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
  data.py             # CV-17 (az) and FLEURS (az_az) loaders
  benchmark.py        # Whisper tiny..large-v3 benchmark
  eval_mms.py         # MMS-1B baseline evaluation
  train_lora.py       # PEFT/LoRA fine-tuning
  analyze_errors.py   # error analysis pipeline
  push_to_hub.py      # Hugging Face upload helper

results/
  benchmark.json
  error_analysis.csv
  transcripts/

models/
  whisper-small-az-lora/
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

### Common Voice 17

Used for:

* LoRA fine-tuning
* additional evaluation

License:

* CC0

### FLEURS Azerbaijani (`az_az`)

Used for:

* multilingual benchmark evaluation
* cross-model comparison

License:

* CC-BY-4.0

---

## Planned Work

* Add Common Voice benchmark tables
* Publish LoRA checkpoints
* Add inference speed / VRAM benchmarks
* Expand linguistic error taxonomy
* Evaluate distillation approaches
* Compare against SeamlessM4T and Canary

---

## License

Code:

* Apache 2.0

Datasets retain their original licenses:

* Common Voice → CC0
* FLEURS → CC-BY-4.0
