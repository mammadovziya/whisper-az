"""LoRA fine-tune Whisper-small on Azerbaijani speech.

Default training data is FLEURS-az `train` (2665 utts) + CV-25 az `train` (215 utts),
shuffled together to ~2880 total. CV-25 alone is too small to fine-tune on. The
eval-during-training set is FLEURS-az `validation` (400 utts) since it's the larger,
cleaner signal. Final test-set WER is measured separately via `src.benchmark` with
the adapter loaded.

Saves the adapter to `models/whisper-small-az-lora/`.

Sanity check (50 steps, ~5 min):
    python -m src.train_lora --max-steps 50 --eval-steps 25 --max-train-samples 200

Full run (5 epochs, ~1-2 h on RTX 5070):
    python -m src.train_lora

Train on a single corpus instead:
    python -m src.train_lora --train-dataset fleurs
    python -m src.train_lora --train-dataset common_voice
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .data import SAMPLING_RATE, load_split

BASE_MODEL = "openai/whisper-small"
OUTPUT_DIR = Path("models/whisper-small-az-lora")


@dataclass
class WhisperDataCollator:
    """Pads input_features (fixed length, just stack) and labels (variable, mask -100)."""

    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch.input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Whisper labels often start with BOS; the model adds it back during decoding shift.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_prepare(processor: WhisperProcessor):
    """Build the per-example feature extractor. Expects `{audio, reference}` columns."""

    def _prep(example: dict) -> dict:
        audio = example["audio"]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=SAMPLING_RATE
        ).input_features[0]
        labels = processor.tokenizer(example["reference"]).input_ids
        return {"input_features": feats, "labels": labels}

    return _prep


def _to_audio_reference(name: str, split: str, max_samples: int | None) -> Dataset:
    """Load a split and project to a uniform `{audio, reference}` schema."""
    ds = load_split(name, split, max_samples=max_samples)
    if name == "fleurs":
        ds = ds.rename_columns({"transcription": "reference"})
    # CV-25 already has 'reference' (renamed inside data.py).
    keep = ["audio", "reference"]
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop)


def load_train_data(choice: str, max_samples: int | None) -> Dataset:
    """Load training data based on --train-dataset selection."""
    if choice == "fleurs":
        return _to_audio_reference("fleurs", "train", max_samples)
    if choice == "common_voice":
        return _to_audio_reference("common_voice", "train", max_samples)
    if choice == "combined":
        f = _to_audio_reference("fleurs", "train", None)
        c = _to_audio_reference("common_voice", "train", None)
        merged = concatenate_datasets([f, c]).shuffle(seed=42)
        if max_samples is not None:
            merged = merged.select(range(min(max_samples, len(merged))))
        return merged
    raise ValueError(f"Unknown --train-dataset {choice!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs (e.g. 50 for a sanity check). -1 means use --epochs.",
    )
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument(
        "--train-dataset",
        choices=["combined", "fleurs", "common_voice"],
        default="combined",
        help="Training data source. 'combined' = FLEURS-az train + CV-25 az train, shuffled.",
    )
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="azerbaijani", task="transcribe"
    )

    train_ds = load_train_data(args.train_dataset, max_samples=args.max_train_samples)
    val_ds = _to_audio_reference("fleurs", "validation", max_samples=args.max_eval_samples)
    print(f"train: {len(train_ds)} samples ({args.train_dataset})")
    print(f"eval:  {len(val_ds)} samples (fleurs/validation)")

    prep = make_prepare(processor)
    train_ds = train_ds.map(prep, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(prep, remove_columns=val_ds.column_names, num_proc=1)

    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "az"
    model.generation_config.task = "transcribe"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # Required for gradient checkpointing to work with PEFT-wrapped models.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=WhisperDataCollator(processor),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    # Save LoRA adapter (small, ~30 MB) plus processor config for downstream loading.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"Saved LoRA adapter to {out.resolve()}")
    print(f"Eval with: python -m src.benchmark --model whisper-small-az-lora --lora-path {out} --dataset all")


if __name__ == "__main__":
    main()
