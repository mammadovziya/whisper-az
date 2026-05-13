"""Error analysis on FLEURS-az: dump per-sentence errors for manual taxonomy, then summarize.

Workflow:

    # 1. Run inference and dump errors to a CSV with an empty `category` column.
    python -m src.analyze_errors dump --model whisper-medium

    # 2. Open results/error_analysis.csv in a spreadsheet, fill the `category` column for
    #    each row using one of the labels in CATEGORIES below.

    # 3. Summarize counts and per-category share.
    python -m src.analyze_errors summarize
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jiwer
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .data import SAMPLING_RATE, iter_samples, load_split, normalize

ERRORS_PATH = Path("results/error_analysis.csv")
SUMMARY_PATH = Path("results/error_summary.txt")

CATEGORIES = [
    "numerals",
    "named_entity",
    "code_switch_ru",
    "code_switch_tr_en",
    "az_letters",  # ə ğ ı ö ş ü ç confusions
    "dialect",
    "audio_quality",
    "other",
]


def transcribe(model_name: str, samples: list, batch_size: int = 4) -> list[str]:
    model_id = f"openai/{model_name}"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    hyps: list[str] = []
    for i in tqdm(range(0, len(samples), batch_size), desc=f"transcribe:{model_name}"):
        batch = samples[i : i + batch_size]
        inputs = processor(
            [s.audio for s in batch],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        )
        feats = inputs.input_features.to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            ids = model.generate(
                feats,
                language="az",
                task="transcribe",
                max_new_tokens=225,
                num_beams=1,
                return_timestamps=False,
            )
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()
    return hyps


def cmd_dump(args: argparse.Namespace) -> None:
    ds = load_split("fleurs", "test", max_samples=args.max_samples)
    samples = list(iter_samples("fleurs", ds))
    hyps = transcribe(args.model, samples, batch_size=args.batch_size)

    rows = []
    for s, h in zip(samples, hyps, strict=True):
        ref_n = normalize(s.reference)
        hyp_n = normalize(h)
        if not ref_n or ref_n == hyp_n:
            continue
        wer = jiwer.wer(ref_n, hyp_n)
        rows.append(
            {
                "sample_id": s.sample_id,
                "reference": s.reference,
                "hypothesis": h,
                "reference_norm": ref_n,
                "hypothesis_norm": hyp_n,
                "wer": round(float(wer), 4),
                "category": "",
            }
        )

    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "sample_id", "reference", "hypothesis", "reference_norm", "hypothesis_norm", "wer", "category"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} error rows to {ERRORS_PATH.resolve()}")
    print(f"Manually fill the `category` column with one of: {', '.join(CATEGORIES)}")


def cmd_summarize(args: argparse.Namespace) -> None:
    if not ERRORS_PATH.exists():
        raise SystemExit(f"{ERRORS_PATH} not found. Run `dump` first.")

    rows: list[dict] = []
    with ERRORS_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("Error CSV is empty.")

    counts: dict[str, int] = {}
    uncategorized = 0
    bad_categories: list[str] = []
    for row in rows:
        cat = row.get("category", "").strip()
        if not cat:
            uncategorized += 1
            continue
        if cat not in CATEGORIES:
            bad_categories.append(cat)
            continue
        counts[cat] = counts.get(cat, 0) + 1

    total = len(rows)
    categorized = total - uncategorized - len(bad_categories)

    lines: list[str] = []
    lines.append(f"Total error sentences: {total}")
    lines.append(f"Categorized: {categorized}")
    lines.append(f"Uncategorized (blank): {uncategorized}")
    if bad_categories:
        lines.append(f"Unknown category labels: {sorted(set(bad_categories))}")
    lines.append("")
    lines.append("Counts by category:")
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        share = n / categorized if categorized else 0.0
        lines.append(f"  {cat:20s} {n:5d}  ({share:5.1%})")

    summary = "\n".join(lines)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote summary to {SUMMARY_PATH.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="Run inference, write error CSV with empty `category` column")
    d.add_argument("--model", default="whisper-medium")
    d.add_argument("--max-samples", type=int, default=None)
    d.add_argument("--batch-size", type=int, default=4)

    sub.add_parser("summarize", help="Read filled-in CSV, print category counts and shares")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "dump":
        cmd_dump(args)
    else:
        cmd_summarize(args)


if __name__ == "__main__":
    main()
