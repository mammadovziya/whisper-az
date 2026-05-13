"""Benchmark Whisper variants on Azerbaijani test sets.

Two backends:
  * HF transformers (fp16) for tiny / base / small / medium and LoRA adapters.
  * faster-whisper (int8_float16) for large-v3 — only way to fit it in 8 GB VRAM.

Output: appends one row per (model, dataset) to results/benchmark.json.

Run:
    python -m src.benchmark --model whisper-tiny --dataset fleurs --max-samples 20
    python -m src.benchmark --model all --dataset all
    python -m src.benchmark --model whisper-small-az-lora --dataset all \\
        --lora-path models/whisper-small-az-lora
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections.abc import Iterator
from pathlib import Path

import jiwer
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .data import (
    DatasetName,
    SAMPLING_RATE,
    iter_samples,
    load_split,
    normalize,
)

RESULTS_PATH = Path("results/benchmark.json")
TRANSCRIPTS_DIR = Path("results/transcripts")

# Maps our short model name -> HF Hub id (for HF backend)
HF_MODEL_IDS: dict[str, str] = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
}

# faster-whisper takes the bare size name
FW_MODEL_IDS: dict[str, str] = {
    "whisper-large-v3": "large-v3",
}

# Default per-model batch size on RTX 5070 8 GB. Override with --batch-size.
DEFAULT_BATCH = {
    "whisper-tiny": 16,
    "whisper-base": 16,
    "whisper-small": 8,
    "whisper-medium": 4,
    "whisper-large-v3": 1,  # faster-whisper transcribes one file at a time
    "whisper-small-az-lora": 8,
}

ALL_MODELS = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v3",
]
ALL_DATASETS: list[DatasetName] = ["fleurs", "common_voice"]


def _batched(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def transcribe_hf(
    model_name: str,
    samples: list,
    batch_size: int,
    lora_path: str | None = None,
) -> list[str]:
    """Transcribe with HF transformers fp16. Returns hypotheses in input order."""
    base_id = HF_MODEL_IDS[model_name] if model_name in HF_MODEL_IDS else "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(base_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
    ).to("cuda")

    if lora_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path).to("cuda")

    model.eval()

    hypotheses: list[str] = []
    for batch in tqdm(list(_batched(samples, batch_size)), desc=f"hf:{model_name}"):
        inputs = processor(
            [s.audio for s in batch],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            pred_ids = model.generate(
                input_features,
                language="az",
                task="transcribe",
                max_new_tokens=225,
                num_beams=1,
                return_timestamps=False,
            )
        hypotheses.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()
    return hypotheses


def _fw_tmp_path(model_name: str, dataset: str, split: str) -> Path:
    return TRANSCRIPTS_DIR / f"_tmp_{model_name}_{dataset}_{split}.json"


def transcribe_fw(model_name: str, samples: list, dataset: str, split: str) -> list[str]:
    """Transcribe with faster-whisper int8_float16 — only path that fits large-v3 in 8 GB.

    Notes on the crash-and-burn we hit here:
      * `del model; torch.cuda.empty_cache()` after the loop silently killed the process
        on Windows + Blackwell + CTranslate2 — segfaults at C level, no Python traceback.
        Hypotheses were complete but never returned.
      * Fix: dump hypotheses to a tmp JSON immediately after the loop. Even if the
        process dies on cleanup, the file survives, and the next run picks it up and
        skips inference entirely.
    """
    from faster_whisper import WhisperModel

    size = FW_MODEL_IDS[model_name]
    model = WhisperModel(size, device="cuda", compute_type="int8_float16")

    hypotheses: list[str] = []
    for s in tqdm(samples, desc=f"fw:{model_name}"):
        segments, _info = model.transcribe(
            s.audio,
            language="az",
            task="transcribe",
            beam_size=1,
            vad_filter=False,
            word_timestamps=False,
        )
        hypotheses.append("".join(seg.text for seg in segments))

    # Crash insurance: persist BEFORE any GPU cleanup that might segfault.
    tmp_path = _fw_tmp_path(model_name, dataset, split)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(hypotheses, ensure_ascii=False), encoding="utf-8")
    print(f"[saved tmp transcripts] {tmp_path}")

    try:
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[warn] non-fatal cleanup error in transcribe_fw: {e}")
    return hypotheses


def evaluate_one(
    model_name: str,
    dataset: DatasetName,
    max_samples: int | None,
    batch_size: int | None,
    lora_path: str | None,
) -> dict:
    split = "test"
    ds = load_split(dataset, split, max_samples=max_samples)
    samples = list(iter_samples(dataset, ds))
    if not samples:
        raise RuntimeError(f"No samples loaded for {dataset}/{split}")

    bs = batch_size or DEFAULT_BATCH.get(model_name, 4)
    t0 = time.time()
    if model_name in FW_MODEL_IDS:
        # Recovery: if a previous run died after writing the tmp file but before
        # computing metrics, pick up the transcripts from disk and skip 17 min of inference.
        tmp_path = _fw_tmp_path(model_name, dataset, split)
        if tmp_path.exists():
            print(f"[recovered] reusing transcripts from {tmp_path}")
            hyps = json.loads(tmp_path.read_text(encoding="utf-8"))
            if len(hyps) != len(samples):
                print(f"[warn] tmp file has {len(hyps)} hyps but {len(samples)} samples — re-running")
                hyps = transcribe_fw(model_name, samples, dataset, split)
        else:
            hyps = transcribe_fw(model_name, samples, dataset, split)
    else:
        hyps = transcribe_hf(model_name, samples, batch_size=bs, lora_path=lora_path)
    elapsed = time.time() - t0

    # Persist raw transcripts BEFORE computing metrics so we never have to redo inference.
    transcripts_path = TRANSCRIPTS_DIR / f"{model_name}_{dataset}_{split}.json"
    save_transcripts(transcripts_path, model_name, dataset, split, samples, hyps)
    print(f"[saved transcripts] {transcripts_path}")

    # Compute WER/CER with jiwer directly (no HF Hub round-trip via `evaluate.load`).
    refs_norm = [normalize(s.reference) for s in samples]
    hyps_norm = [normalize(h) for h in hyps]
    # jiwer.wer/cer raise on empty references — guard by replacing empty refs with a single space.
    refs_norm = [r if r else " " for r in refs_norm]
    wer = float(jiwer.wer(refs_norm, hyps_norm))
    cer = float(jiwer.cer(refs_norm, hyps_norm))

    return {
        "model": model_name,
        "dataset": dataset,
        "split": split,
        "n_samples": len(samples),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "elapsed_seconds": round(elapsed, 1),
        "lora_path": lora_path,
        "transcripts_path": str(transcripts_path),
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def save_transcripts(
    path: Path,
    model_name: str,
    dataset: str,
    split: str,
    samples: list,
    hyps: list[str],
) -> None:
    """Write per-sample (sample_id, reference, hypothesis) triples to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "dataset": dataset,
        "split": split,
        "n_samples": len(samples),
        "samples": [
            {"sample_id": s.sample_id, "reference": s.reference, "hypothesis": h}
            for s, h in zip(samples, hyps, strict=True)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_result(row: dict, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists() and path.stat().st_size > 0:
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.append(row)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help=f"One of {ALL_MODELS} or 'whisper-small-az-lora' or 'all'",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=[*ALL_DATASETS, "all"],
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--lora-path",
        default=None,
        help="Path to PEFT LoRA adapter (only used when --model is whisper-small-az-lora)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = ALL_MODELS if args.model == "all" else [args.model]
    datasets: list[DatasetName] = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for model_name in models:
        for dataset in datasets:
            print(f"\n=== {model_name} on {dataset} ===")
            row = evaluate_one(
                model_name=model_name,
                dataset=dataset,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                lora_path=args.lora_path if model_name == "whisper-small-az-lora" else None,
            )
            print(json.dumps(row, indent=2, ensure_ascii=False))
            append_result(row)


if __name__ == "__main__":
    main()
