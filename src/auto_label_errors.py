"""Auto-label `results/error_analysis.csv` rows with error categories.

Rule-based, text-only labeling. The taxonomy from `analyze_errors.py`:

    numerals          ref contains digits
    code_switch_ru    ref contains Cyrillic
    named_entity      hypothesis has a non-sentence-initial capitalized word
                      whose case-folded form isn't found in the reference
                      (model heard a name and transcribed it differently)
    az_letters        ref has more Azerbaijani-specific letters (ə ğ ı ö ş ü ç)
                      than the normalized hypothesis (model dropped diacritics)
    code_switch_tr_en | not reliably detectable from text alone
    dialect           |
    audio_quality     |
    other             everything that didn't trigger a higher-priority rule

`audio_quality` and `dialect` need the audio (or native-speaker review).
`code_switch_tr_en` needs a word-list / language ID step we don't have.
Those three are not auto-assigned; they remain available for manual override.

Rules are applied in the priority order above; the first match wins. Treat
this as a first-pass distribution, not ground truth. Run after `analyze_errors.py
dump` and before `analyze_errors.py summarize`.

Usage:

    python -m src.auto_label_errors
    python -m src.analyze_errors summarize
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ERRORS_PATH = Path("results/error_analysis.csv")

# Azerbaijani-specific letters (lowercase forms; ref/hyp_norm are already lowercased).
AZ_SPECIFIC = set("əğıöşüç")

DIGIT_RE = re.compile(r"\d")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
# Non-sentence-initial capitalized token. Strip surrounding punctuation, then check
# that first char is uppercase letter AND the token has at least one lowercase letter
# (rules out single-letter capitalizations and pure-punctuation).
WORD_SPLIT_RE = re.compile(r"\s+")
PUNCT_STRIP = '.,!?:;"\'()[]{}—–-«»„"'


def _has_named_entity(hypothesis: str, reference_norm: str) -> bool:
    """Heuristic: a capitalized non-initial token in the hypothesis whose
    case-folded form does NOT appear as a whole word in the reference. This
    catches the common pattern where the model transcribes a proper noun
    differently from how the reference spells it."""
    tokens = WORD_SPLIT_RE.split(hypothesis.strip())
    if len(tokens) <= 1:
        return False
    ref_words = set(reference_norm.split())
    for tok in tokens[1:]:  # skip first token (sentence-initial cap is normal)
        clean = tok.strip(PUNCT_STRIP)
        if not clean or not clean[0].isupper():
            continue
        # has a lowercase letter somewhere → looks like a Name rather than ALL-CAPS noise
        if not any(c.islower() for c in clean):
            continue
        if clean.casefold() not in ref_words:
            return True
    return False


def _has_lost_az_letters(reference_norm: str, hypothesis_norm: str) -> bool:
    """ref has more az-specific letters than hyp → diacritic loss."""
    ref_count = sum(1 for c in reference_norm if c in AZ_SPECIFIC)
    hyp_count = sum(1 for c in hypothesis_norm if c in AZ_SPECIFIC)
    return ref_count > 0 and ref_count > hyp_count


def categorize(row: dict[str, str]) -> str:
    ref = row.get("reference", "") or ""
    hyp = row.get("hypothesis", "") or ""
    ref_norm = row.get("reference_norm", "") or ""
    hyp_norm = row.get("hypothesis_norm", "") or ""

    # Priority 1: numerals — digits in the reference essentially always cause errors
    if DIGIT_RE.search(ref):
        return "numerals"

    # Priority 2: Russian code-switching — Cyrillic in the reference
    if CYRILLIC_RE.search(ref):
        return "code_switch_ru"

    # Priority 3: named entity confusion — heuristic on hyp capitalization
    if _has_named_entity(hyp, ref_norm):
        return "named_entity"

    # Priority 4: dropped Azerbaijani diacritics
    if _has_lost_az_letters(ref_norm, hyp_norm):
        return "az_letters"

    return "other"


def main() -> None:
    if not ERRORS_PATH.exists():
        raise SystemExit(f"{ERRORS_PATH} not found. Run `analyze_errors dump` first.")
    with ERRORS_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Error CSV has no rows.")

    counts: dict[str, int] = {}
    overwritten = 0
    for row in rows:
        prev = row.get("category", "").strip()
        new = categorize(row)
        if prev and prev != new:
            overwritten += 1
        row["category"] = new
        counts[new] = counts.get(new, 0) + 1

    # Write back, preserving original column order.
    fieldnames = list(rows[0].keys())
    with ERRORS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"Labeled {total} rows. Overwrote {overwritten} previously-labeled rows.")
    print()
    print("Auto-label distribution:")
    for cat in sorted(counts, key=lambda c: -counts[c]):
        share = counts[cat] / total
        print(f"  {cat:20s} {counts[cat]:5d}  ({share:5.1%})")


if __name__ == "__main__":
    main()
