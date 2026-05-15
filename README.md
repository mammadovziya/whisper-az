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

| Model | Trainable params | FLEURS-az WER | CV-25 az WER |
|---|---:|---:|---:|
| Whisper Small | 244M | 51.7% | 62.8% |
| **Whisper Small + our LoRA** | **244M + 13M** | **35.0%** | **40.7%** |
| Whisper Medium *(3× larger)* | 769M | 34.3% | 42.1% |
| Whisper Large-v3 *(6× larger)* | 1550M | 21.7% | 26.5% |

Trained on a single laptop GPU (RTX 5070, 8 GB VRAM, 58 min). The adapter matches `medium` on FLEURS and **beats it on CV-25** — with a base model three times smaller.

---

## Quickstart

```bash
git clone https://github.com/mammadovziya/whisper-az.git
cd whisper-az

# Torch with Blackwell support (RTX 50-series). Swap cu128 for cu121 on older GPUs.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Smoke test: ~5 min, downloads whisper-tiny and 20 FLEURS-az samples
python -m src.benchmark --model whisper-tiny --dataset fleurs --max-samples 20
```

---

## Three findings worth sharing

**1. The scaling curve is steep.** Going from `whisper-tiny` to `whisper-large-v3` cuts WER by ~5× on Azerbaijani. The first useful jump is `base → small`; below that, both `tiny` and `base` hallucinate so badly that WER exceeds 100%.

**2. MMS-1B's 9 MB Azerbaijani adapter is genuinely competitive.** On both test sets it lands within 2 WER points of `whisper-large-v3` — and wins on CER (it makes smaller, character-level errors). For offline use on weak hardware, MMS is the strongest baseline by far.

**3. A 51 MB LoRA on whisper-small matches a 3× larger checkpoint.** Five epochs on ~2,900 training samples close most of the small→medium gap. You get medium-tier accuracy at small-tier inference cost — and you can ship the adapter, not the whole checkpoint.

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
| Whisper Small + LoRA *(ours)* | 244M + 13M | **40.7%** | 12.2% |
| Whisper Medium | 769M | 42.1% | 12.2% |
| Whisper Large-v3 | 1550M | **26.5%** | 6.9% |
| MMS-1B (azj-latin) | 1B + 9M | 28.3% | **6.3%** |

Source-of-truth numbers and per-sample transcripts live in `results/`.

---

## Example transcription

Reference:

```text
Mən sabah universitetə gedəcəyəm.
```

Whisper Tiny (Turkish interference):

```text
Ben sabah üniversiteye gideceğim.
```

Whisper Large-v3 (correct):

```text
Mən sabah universitetə gedəcəyəm.
```

Smaller models repeatedly fall back to Turkish orthography, dropping the Azerbaijani-specific letters `ə ğ ı ö ş ü ç`. The error taxonomy in `results/error_analysis.csv` quantifies this: **19% of whisper-medium's FLEURS-az errors are pure diacritic loss**, and another **24% are mistranscribed proper nouns**.

---

## Error taxonomy

All 923 FLEURS-az errors from whisper-medium, auto-labeled by `src/auto_label_errors.py`:

| Category | Count | Share |
|---|---:|---:|
| Other (residual phonetic / morphological) | 316 | 34.2% |
| Named entities (proper nouns) | 222 | 24.1% |
| Numerals (digits / numbers) | 211 | 22.9% |
| Azerbaijani-specific letters lost | 174 | 18.9% |
| Russian / TR-EN code-switching | 0 | 0.0% |

Two-thirds of the errors fall into three machine-identifiable buckets — names, numbers, and dropped diacritics. The remaining 34% are subtle phonetic confusions that text rules can't classify.

---

## LoRA fine-tuning details

* **Base:** `openai/whisper-small` (244M)
* **LoRA:** r=32, α=64, dropout=0.05, targets `q/k/v/o_proj` + `fc1/fc2`
* **Trainable:** 12.97M params (5.09% of base), 51 MB on disk
* **Training data:** FLEURS-az `train` (2,665) + CV-25 az `train` (215), shuffled — 2,880 total
* **Eval-during-training:** FLEURS-az `validation` (400)
* **Hparams:** 5 epochs, effective batch 16, lr 1e-4, warmup 100, fp16, gradient checkpointing
* **Compute:** single RTX 5070 Laptop, **58 min wall time** (900 steps)
* **Loss:** eval_loss 1.40 → 0.51 on FLEURS validation

Relative WER reduction over baseline `whisper-small`:

* **FLEURS-az: −32%** (51.7 → 35.0)
* **CV-25 az: −35%** (62.8 → 40.7)

The full training loss curve is in `results/train_lora_history.txt`.

---

## Reproducing everything

```bash
# Full whisper benchmark on both datasets (~2 h on RTX 5070)
python -m src.benchmark --model all --dataset all

# MMS-1B baseline
python -m src.eval_mms --dataset all

# LoRA fine-tune (~1 h)
python -m src.train_lora

# Evaluate the fine-tuned adapter
python -m src.benchmark --model whisper-small-az-lora \
  --lora-path models/whisper-small-az-lora --dataset all

# Error analysis pipeline
python -m src.analyze_errors dump --model whisper-medium
python -m src.auto_label_errors
python -m src.analyze_errors summarize
```

---

## Project structure

```text
src/
  data.py               # FLEURS (az_az, HF) + CV-25 (az, local MDC tarball) loaders
  benchmark.py          # Whisper benchmark (HF for tiny..medium, faster-whisper for large-v3)
  eval_mms.py           # MMS-1B baseline (azj-script_latin adapter)
  train_lora.py         # PEFT/LoRA fine-tuning on whisper-small
  analyze_errors.py     # error dump + per-category summary
  auto_label_errors.py  # rule-based first-pass error categorization
  push_to_hub.py        # Hugging Face upload helper

results/
  benchmark.json                  # WER/CER table (model × dataset)
  error_analysis.csv              # 923 per-sentence errors + auto-labels
  transcripts/                    # per-sample (ref, hyp) for every run
  train_lora_history.txt          # LoRA training loss curve

data/cv25-az/                     # extracted MDC tarball (gitignored)
models/whisper-small-az-lora/     # fine-tuned LoRA adapter (gitignored)
```

---

## Datasets

**FLEURS-az** (`google/fleurs`, config `az_az`). Clean read speech, ~10 hours total. Primary benchmark. CC-BY-4.0. Loads directly from HuggingFace Hub.

**Common Voice 25.0 az** (Mozilla Data Collective). Crowd-sourced speech, much noisier than FLEURS. Cross-domain evaluation. Only 215 train + 93 dev + 126 test utterances — `az` is genuinely under-represented even on CV. CC0.

> ⚠️ **Common Voice distribution moved in Oct 2025.** Mozilla now serves CV exclusively through [Mozilla Data Collective](https://mozilladatacollective.com/), not HuggingFace Hub. To reproduce CV-25 results: sign up at MDC, accept the dataset terms, download the `az` tarball, and extract it to `data/cv25-az/`. The loader in `src/data.py` reads directly from the extracted directory — no API key needed at runtime.

---

## Hardware

Developed on a single RTX 5070 Laptop GPU (8 GB VRAM, Blackwell `sm_120`). Everything in this repo fits — `large-v3` inference uses `faster-whisper` int8 to stay under 4 GB; LoRA training uses fp16 + gradient checkpointing.

> RTX 50-series GPUs require **`cu128`** or newer PyTorch wheels. The `cu126` build will install but produces `sm_120 not compatible` warnings and won't actually run on the GPU.

---

## Roadmap

* [ ] Publish the LoRA adapter to HuggingFace Hub
* [ ] Native-speaker pass on ~50 worst `other`-bucket errors
* [ ] Inference speed / VRAM benchmarks (RTF, peak memory)
* [ ] Distillation: can we get a 39M-param student from large-v3?
* [ ] Compare against SeamlessM4T and NVIDIA Canary

---

## License

Code: **Apache 2.0**. Datasets retain their original licenses (FLEURS → CC-BY-4.0; Common Voice → CC0). The fine-tuned LoRA adapter, when published, will be Apache 2.0.
