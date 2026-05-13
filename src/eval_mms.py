"""MMS-1B (`facebook/mms-1b-all`) baseline on Azerbaijani test sets.

MMS uses per-language adapters; we load the `azj` (North Azerbaijani) adapter, which is
the variant spoken in Azerbaijan (Republic of) and matches FLEURS-az / Common Voice-az.

Run:
    python -m src.eval_mms --dataset all
    python -m src.eval_mms --dataset fleurs --max-samples 50
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
from transformers import AutoProcessor, Wav2Vec2ForCTC

from .benchmark import RESULTS_PATH, TRANSCRIPTS_DIR, append_result, save_transcripts
from .data import (
    DatasetName,
    SAMPLING_RATE,
    iter_samples,
    load_split,
    normalize,
)

MMS_MODEL_ID = "facebook/mms-1b-all"
# MMS adapter names include script. North Azerbaijani (Republic of Azerbaijan) uses Latin script.
# (The other available variant is `azj-script_cyrillic`, used in some diaspora communities.)
MMS_LANG_CODE = "azj-script_latin"


def _batched(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def transcribe_mms(samples: list, batch_size: int = 4) -> list[str]:
    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID, torch_dtype=torch.float16).to("cuda")

    processor.tokenizer.set_target_lang(MMS_LANG_CODE)
    model.load_adapter(MMS_LANG_CODE)
    model.eval()

    hypotheses: list[str] = []
    for batch in tqdm(list(_batched(samples, batch_size)), desc=f"mms-1b/{MMS_LANG_CODE}"):
        inputs = processor(
            [s.audio for s in batch],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to("cuda", dtype=torch.float16)
        attention_mask = inputs.attention_mask.to("cuda")
        with torch.inference_mode():
            logits = model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hypotheses.extend(processor.batch_decode(pred_ids))

    del model
    torch.cuda.empty_cache()
    return hypotheses


def evaluate_mms(dataset: DatasetName, max_samples: int | None, batch_size: int) -> dict:
    split = "test"
    ds = load_split(dataset, split, max_samples=max_samples)
    samples = list(iter_samples(dataset, ds))
    if not samples:
        raise RuntimeError(f"No samples loaded for {dataset}/{split}")

    t0 = time.time()
    hyps = transcribe_mms(samples, batch_size=batch_size)
    elapsed = time.time() - t0

    model_label = f"mms-1b-all/{MMS_LANG_CODE}"
    transcripts_path = TRANSCRIPTS_DIR / f"mms-1b-all-{MMS_LANG_CODE}_{dataset}_{split}.json"
    save_transcripts(transcripts_path, model_label, dataset, split, samples, hyps)
    print(f"[saved transcripts] {transcripts_path}")

    refs_norm = [normalize(s.reference) for s in samples]
    hyps_norm = [normalize(h) for h in hyps]
    refs_norm = [r if r else " " for r in refs_norm]

    wer = float(jiwer.wer(refs_norm, hyps_norm))
    cer = float(jiwer.cer(refs_norm, hyps_norm))

    return {
        "model": model_label,
        "dataset": dataset,
        "split": split,
        "n_samples": len(samples),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "elapsed_seconds": round(elapsed, 1),
        "lora_path": None,
        "transcripts_path": str(transcripts_path),
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["fleurs", "common_voice", "all"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets: list[DatasetName] = (
        ["fleurs", "common_voice"] if args.dataset == "all" else [args.dataset]
    )
    for dataset in datasets:
        print(f"\n=== mms-1b-all/{MMS_LANG_CODE} on {dataset} ===")
        row = evaluate_mms(dataset, args.max_samples, args.batch_size)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        append_result(row, RESULTS_PATH)


if __name__ == "__main__":
    main()
