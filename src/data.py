"""Azerbaijani ASR dataset loaders and shared text normalization.

Two datasets:
  * `fleurs`        -> google/fleurs, config az_az  (clean read speech, ~10h, our primary benchmark)
  * `common_voice`  -> Common Voice 25.0 az, loaded from a LOCAL extracted tarball at
                      `data/cv25-az/cv-corpus-25.0-2026-03-09/az/`. As of Oct 2025
                      Mozilla distributes CV exclusively through Mozilla Data Collective;
                      the HF Hub repos are not programmatically accessible.

All audio is decoded at 16 kHz mono. References are returned raw — callers must apply
`normalize()` before computing WER/CER so that all numbers are computed on identical
normalization (the same convention used in the Whisper paper).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import csv

import numpy as np
from datasets import Audio, Dataset, load_dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

SAMPLING_RATE = 16_000
HF_CACHE_DIR = os.environ.get("HF_CACHE", "./hf_cache")

# CV-25 local layout (extracted from the MDC tarball).
CV25_ROOT = Path(os.environ.get("CV25_ROOT", "data/cv25-az/cv-corpus-25.0-2026-03-09/az"))
CV25_SPLIT_FILE = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}

DatasetName = Literal["fleurs", "common_voice"]
Split = Literal["train", "validation", "test"]

_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    """Whisper's multilingual basic normalizer. Apply to both references and hypotheses."""
    return _normalizer(text)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    audio: np.ndarray  # mono float32 at 16 kHz
    reference: str     # raw transcript, un-normalized


def _load_cv25_local(split: Split) -> Dataset:
    """Load Common Voice 25.0 az from the extracted MDC tarball at CV25_ROOT.

    We avoid `Dataset.from_pandas` here because modern pandas encodes string columns
    as pyarrow `large_string`, which `Audio.cast_storage` refuses to cast
    (`ArrowNotImplementedError: Unsupported cast from large_string to struct`).
    Going through `Dataset.from_dict` with Python str values produces plain `string`
    columns, which cast to Audio cleanly. CV TSVs are small (<1 MB), so reading via
    the csv module is plenty fast.
    """
    if split not in CV25_SPLIT_FILE:
        raise ValueError(f"Unknown CV split: {split!r}")
    tsv_path = CV25_ROOT / CV25_SPLIT_FILE[split]
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Expected CV-25 split file at {tsv_path}. "
            "Did you extract the MDC tarball into data/cv25-az/?"
        )
    audio: list[str] = []
    reference: list[str] = []
    path: list[str] = []
    client_id: list[str] = []
    sentence_id: list[str] = []
    with tsv_path.open(encoding="utf-8", newline="") as f:
        # quoting=QUOTE_NONE — CV transcripts contain bare quotes that aren't pair-delimiters.
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            audio.append(str((CV25_ROOT / "clips" / row["path"]).resolve()))
            reference.append(row.get("sentence", "") or "")
            path.append(row["path"])
            client_id.append(row.get("client_id", "") or "")
            sentence_id.append(row.get("sentence_id", "") or "")
    ds = Dataset.from_dict({
        "audio": audio,
        "reference": reference,
        "path": path,
        "client_id": client_id,
        "sentence_id": sentence_id,
    })
    return ds


def load_split(
    name: DatasetName,
    split: Split,
    max_samples: int | None = None,
) -> Dataset:
    """Load an Azerbaijani audio split with audio resampled to 16 kHz mono."""
    if name == "fleurs":
        ds = load_dataset(
            "google/fleurs",
            "az_az",
            split=split,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True,
        )
    elif name == "common_voice":
        ds = _load_cv25_local(split)
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Expected 'fleurs' or 'common_voice'.")

    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def reference_of(name: DatasetName, row: dict) -> str:
    if name == "fleurs":
        return row["transcription"]
    # common_voice — we renamed `sentence` to `reference` during load
    return row["reference"]


def sample_id_of(name: DatasetName, row: dict) -> str:
    if name == "fleurs":
        return f"fleurs-{row['id']}"
    # common_voice (CV-25)
    return f"cv25-{row['path']}"


def iter_samples(name: DatasetName, ds: Dataset) -> Iterator[Sample]:
    """Iterate as `Sample` objects — convenient when you don't need HF batching."""
    for row in ds:
        yield Sample(
            sample_id=sample_id_of(name, row),
            audio=np.asarray(row["audio"]["array"], dtype=np.float32),
            reference=reference_of(name, row),
        )
