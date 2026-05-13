"""Push the LoRA adapter (and a generated model card) to the HuggingFace Hub.

Prerequisites:
    huggingface-cli login

Run:
    python -m src.push_to_hub --repo ziyamammadov/whisper-small-az-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from .benchmark import RESULTS_PATH


def build_model_card(
    repo_id: str,
    adapter_dir: Path,
    results_path: Path = RESULTS_PATH,
) -> str:
    """Generate a model card containing benchmark numbers from results/benchmark.json."""
    rows: list[dict] = []
    if results_path.exists() and results_path.stat().st_size > 0:
        rows = json.loads(results_path.read_text(encoding="utf-8"))

    base_rows = [r for r in rows if r["model"] == "whisper-small"]
    lora_rows = [r for r in rows if r["model"] == "whisper-small-az-lora"]

    def _format_table(label: str, included: list[dict]) -> str:
        if not included:
            return f"_{label}: no rows in benchmark.json yet._\n"
        out = ["| dataset | n | WER | CER |", "|---|---:|---:|---:|"]
        for r in included:
            out.append(
                f"| {r['dataset']} | {r['n_samples']} | {r['wer']:.4f} | {r['cer']:.4f} |"
            )
        return "\n".join(out)

    return f"""---
language:
  - az
license: apache-2.0
base_model: openai/whisper-small
library_name: peft
tags:
  - whisper
  - lora
  - peft
  - automatic-speech-recognition
  - azerbaijani
datasets:
  - mozilla-foundation/common_voice_17_0
  - google/fleurs
metrics:
  - wer
  - cer
---

# {repo_id.split("/")[-1]}

LoRA adapter for [openai/whisper-small](https://huggingface.co/openai/whisper-small) fine-tuned on the
Azerbaijani (`az`) split of Common Voice 17. Part of an effort to document and improve Whisper's
performance on Azerbaijani — see the [paper / writeup](https://github.com/{repo_id.split("/")[0]}/azerbaijani-asr) for the full benchmark
across all Whisper sizes plus MMS-1B baseline and a manual error taxonomy.

## Results

### whisper-small-az-lora (this adapter)

{_format_table("LoRA results", lora_rows)}

### whisper-small (base, for comparison)

{_format_table("Base results", base_rows)}

WER and CER are computed after applying `transformers.models.whisper.tokenization_whisper.BasicTextNormalizer`
to both references and hypotheses — same convention as the Whisper paper.

## Usage

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

base = "openai/whisper-small"
adapter = "{repo_id}"

processor = WhisperProcessor.from_pretrained(base)
model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, adapter).to("cuda")
model.eval()

# audio: 1-D float32 numpy array at 16 kHz
inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
features = inputs.input_features.to("cuda", dtype=torch.float16)

with torch.inference_mode():
    pred_ids = model.generate(features, language="az", task="transcribe", max_new_tokens=225)
print(processor.batch_decode(pred_ids, skip_special_tokens=True)[0])
```

## Training

PEFT LoRA, `r=32`, `α=64`, dropout `0.05`. Targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`.
Train: Common Voice 17 az `train`. Eval: Common Voice 17 az `validation`. 5 epochs, lr `1e-4`,
warmup 100 steps, fp16, gradient checkpointing, batch size 8 × accumulation 2. Trained on a single
RTX 5070 Laptop GPU (8 GB VRAM).

## Limitations

- Trained on Common Voice — read speech in (mostly) clean conditions. Performance on
  spontaneous, noisy, or accented speech is not guaranteed.
- Common Voice transcripts are crowd-sourced and contain transcription noise.
- Single-seed training; no hyperparameter sweep beyond defaults.
- LoRA adapter requires the base `openai/whisper-small` weights at inference.

## License

Apache 2.0 (matches `openai/whisper-small`). Common Voice is CC0; FLEURS is CC-BY-4.0.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        required=True,
        help="HF Hub repo id, e.g. `ziyamammadov/whisper-small-az-lora`",
    )
    p.add_argument("--adapter-dir", default="models/whisper-small-az-lora")
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--write-card-only",
        action="store_true",
        help="Write README.md to adapter dir without uploading.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter dir not found: {adapter_dir.resolve()}")

    card = build_model_card(args.repo, adapter_dir)
    (adapter_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"Wrote model card to {(adapter_dir / 'README.md').resolve()}")

    if args.write_card_only:
        return

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=args.repo)
    print(f"Pushed {adapter_dir} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
