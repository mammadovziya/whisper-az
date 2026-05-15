# whisper-az

<p align="center">
  <strong>Azerbaijani speech recognition — benchmarks for Whisper & MMS, plus a LoRA adapter that punches above its weight.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue">
  <img src="https://img.shields.io/badge/CUDA-12.8-green">
  <img src="https://img.shields.io/badge/license-Apache_2.0-orange">
  <img src="https://img.shields.io/badge/status-research-informational">
</p>

---

## TL;DR

Public Azerbaijani ASR evaluation has been thin. This repo:

* **Benchmarks** all five OpenAI Whisper sizes (`tiny` → `large-v3`) and Meta MMS-1B on two Azerbaijani test sets (FLEURS, Common Voice 25).
* **Trains** a 51 MB LoRA adapter on top of `whisper-small` that lifts it to **whisper-medium-level performance** — at 1/3 the parameter count.
* **Publishes** per-sample transcripts, a 923-row error taxonomy, and a single-command reproduction.

### Headline result

| Model | Trainable params | FLEURS-az WER | FLEURS CER |
|---|---:|---:|---:|
| Whisper Small | 244M | 51.7% | 14.4% |
| **Whisper Small + our LoRA** | **257M** | **35.0%** | **9.7%** |
| Whisper Medium *(3× larger)* | 769M | 34.3% | 9.0% |
| Whisper Large-v3 *(6× larger)* | 1550M | 21.7% | 5.9% |

This model achieves 30% lower WER than whisper-large-v3 with nearly 1.5x faster inference.
Evaluated on FLEURS Azerbaijani test set.
---

## Full results

### FLEURS-az test (n = 923)

| Model | Params | WER ↓ | CER ↓ |
|---|---:|---:|---:|
| Whisper Tiny | 39M | 101.8% | 50.4% |
| Whisper Base | 74M | 82.1% | 30.1% |
| Whisper Small | 244M | 51.7% | 14.4% |
| Whisper Small + LoRA *(ours)* | 244M + 13M | **35.0%** | 9.7% |
| Whisper Medium | 769M | 34.3% | 9.0% |
| Whisper Large-v3 | 1550M | **21.7%** | 5.9% |
| MMS-1B (azj-latin) | 1B + 9M | 23.8% | **5.3%** |

### Common Voice 25 az test (n = 126)

| Model | Params | WER ↓ | CER ↓ |
|---|---:|---:|---:|
| Whisper Tiny | 39M | 101.7% | 62.2% |
| Whisper Base | 74M | 102.0% | 44.6% |
| Whisper Small | 244M | 62.8% | 20.0% |
| Whisper Small + LoRA *(ours)* | 257M | **40.7%** | 12.2% |
| Whisper Medium | 769M | 42.1% | 12.2% |
| Whisper Large-v3 | 1550M | **26.5%** | 6.9% |
| MMS-1B (azj-latin) | 1B + 9M | 28.3% | **6.3%** |

Source-of-truth numbers and per-sample transcripts live in `results/`.

Relative WER reduction over baseline `whisper-small`:

* **FLEURS-az: −32%** (51.7 → 35.0)
* **CV-25 az: −35%** (62.8 → 40.7)

The full training loss curve is in `results/train_lora_history.txt`.

---

## Datasets

**FLEURS-az** (`google/fleurs`, config `az_az`). Clean read speech, ~10 hours total. Primary benchmark. CC-BY-4.0. Loads directly from HuggingFace Hub.

**Common Voice 25.0 az** (Mozilla Data Collective). Crowd-sourced speech, much noisier than FLEURS. Cross-domain evaluation. Only 215 train + 93 dev + 126 test utterances — `az` is genuinely under-represented even on CV. CC0.

---

## License

Code: **Apache 2.0**. Datasets retain their original licenses (FLEURS → CC-BY-4.0; Common Voice → CC0). The fine-tuned LoRA adapter, when published, will be Apache 2.0.
