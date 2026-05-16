# Azerbaijani Whisper

OpenAI Whisper benchmarked on Azerbaijani speech, with a parameter-efficient LoRA adapter trained on top of `whisper-small` that closes most of the gap to `whisper-medium`. Evaluated on FLEURS-az and Common Voice 25 az, with per-sample transcripts and a 923-row error taxonomy published alongside.

## Performance

Evaluation on the FLEURS Azerbaijani test split (n = 923).

| Model | Params | WER | CER |
|---|---:|---:|---:|
| Whisper Small | 244M | 51.7% | 14.4% |
| **Whisper Small + LoRA (ours)** | **244M + 13M** | **35.0%** | **9.7%** |
| Whisper Medium | 769M | 34.3% | 9.0% |
| Whisper Large-v3 | 1550M | 21.7% | 5.9% |
| MMS-1B (azj-latin) | 1B + 9M | 23.8% | 5.3% |

A 51 MB LoRA adapter (only 5.1% of the base parameters trained) brings `whisper-small` to medium-tier accuracy — at one third the parameter count and a fraction of the inference cost. Training took 58 minutes on a single RTX 5070 laptop GPU.

## Usage

```python
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="az", task="transcribe"
)
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", torch_dtype=torch.float16
).to("cuda")
model = PeftModel.from_pretrained(base, "models/whisper-small-az-lora").to("cuda")
model.eval()

# audio: mono float32 array at 16 kHz
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to("cuda", dtype=torch.float16)

with torch.inference_mode():
    pred_ids = model.generate(
        input_features, language="az", task="transcribe", max_new_tokens=225
    )

text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
```

> Always force `language="az"` and `task="transcribe"` in `generate()`. Without them Whisper auto-detects the language and frequently routes Azerbaijani audio to Turkish, producing systematic orthographic errors.

## Requirements

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

RTX 50-series (Blackwell, `sm_120`) GPUs require `cu128` or newer PyTorch wheels.

## Which model to choose?

| Scenario | Recommendation | FLEURS WER | Footprint |
|---|---|---:|---|
| Maximum accuracy | `whisper-large-v3` | 21.7% | 1.55B params, ~10 GB VRAM |
| Best on weak hardware | MMS-1B (azj-latin) | 23.8% | 1B base + 9 MB adapter |
| Compact deployment | **Whisper Small + LoRA** | **35.0%** | 244M base + 51 MB adapter |
| OpenAI baseline | `whisper-medium` | 34.3% | 769M, no fine-tuning |

If you need the smallest deployable footprint with reasonable accuracy, use the LoRA adapter. If accuracy is the only thing that matters, use `whisper-large-v3`.

## Benchmark Details

All numbers computed with the multilingual Whisper basic text normalizer, beam size 1, fp16 inference (int8 for `large-v3`). Source-of-truth values and per-sample transcripts live in `results/`.

| Model | Params | FLEURS WER | FLEURS CER | CV-25 WER | CV-25 CER |
|---|---:|---:|---:|---:|---:|
| Whisper Tiny | 39M | 101.8% | 50.4% | 101.7% | 62.2% |
| Whisper Base | 74M | 82.1% | 30.1% | 102.0% | 44.6% |
| Whisper Small | 244M | 51.7% | 14.4% | 62.8% | 20.0% |
| **Whisper Small + LoRA** | **244M + 13M** | **35.0%** | **9.7%** | **40.7%** | **12.2%** |
| Whisper Medium | 769M | 34.3% | 9.0% | 42.1% | 12.2% |
| Whisper Large-v3 | 1550M | 21.7% | 5.9% | 26.5% | 6.9% |
| MMS-1B (azj-latin) | 1B + 9M | 23.8% | 5.3% | 28.3% | 6.3% |

### LoRA training

| | |
|---|---|
| Base model | `openai/whisper-small` |
| LoRA config | r=32, α=64, dropout=0.05; targets `q/k/v/o_proj` + `fc1/fc2` |
| Trainable params | 12.97M (5.09% of base), 51 MB on disk |
| Training data | FLEURS-az train (2,665) + Common Voice 25 az train (215), shuffled |
| Eval-during-training | FLEURS-az validation (400) |
| Hyperparameters | 5 epochs, batch 8 × accum 2, lr 1e-4, warmup 100, fp16, gradient checkpointing |
| Compute | Single RTX 5070 Laptop, 8 GB VRAM, 58 min wall time |
| Final eval loss | 1.40 → 0.51 on FLEURS validation |

### Error taxonomy

All 923 `whisper-medium` errors on FLEURS-az, auto-categorized by text-based rules.

| Category | Count | Share |
|---|---:|---:|
| Named entities | 222 | 24.1% |
| Numerals | 211 | 22.9% |
| Azerbaijani-specific letters dropped (`ə ğ ı ö ş ü ç`) | 174 | 18.9% |
| Residual (phonetic / morphological) | 316 | 34.2% |

Two thirds of all errors fall into three machine-identifiable buckets — names, numbers, and lost diacritics.

### Reproducing

```bash
# Full benchmark (both datasets, all Whisper sizes + MMS, ~2 h on RTX 5070)
python -m src.benchmark --model all --dataset all
python -m src.eval_mms --dataset all

# LoRA fine-tune + evaluation (~1 h)
python -m src.train_lora
python -m src.benchmark --model whisper-small-az-lora \
  --lora-path models/whisper-small-az-lora --dataset all

# Error analysis
python -m src.analyze_errors dump --model whisper-medium
python -m src.auto_label_errors
python -m src.analyze_errors summarize
```

### Datasets

- **FLEURS-az** — `google/fleurs`, config `az_az`. Clean read speech, ~10 hours. CC-BY-4.0. Loaded directly from HuggingFace Hub.
- **Common Voice 25.0 az** — crowd-sourced speech, 215 train / 93 dev / 126 test. CC0. Distributed exclusively through [Mozilla Data Collective](https://mozilladatacollective.com/) since October 2025. Sign up, accept the dataset terms, download the `az` tarball, and extract to `data/cv25-az/`.

## License

Apache 2.0
