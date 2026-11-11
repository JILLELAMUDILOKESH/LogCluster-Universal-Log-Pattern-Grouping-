"""Unsupervised log normalization and similarity clustering."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


_VALUE_PATTERNS = (
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b", re.I), "<TIMESTAMP>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<DATE>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"), "<TIME>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I), "<UUID>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "<ID>"),
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|sec|seconds|bytes|kb|mb|gb)?\b", re.I), "<NUM>"),
)


@dataclass(frozen=True)
class PatternGroup:
    """A discovered pattern and its most representative original log line."""

    template: str
    example: str
    count: int


def normalize_line(line: str) -> str:
    """Replace common dynamic log values with placeholders."""
    normalized = line
    for pattern, replacement in _VALUE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def read_dataset(file_name: str, content: bytes) -> pd.DataFrame:
    """Read a supported structured dataset into a table."""
    suffix = Path(file_name).suffix.casefold()
    stream = BytesIO(content)
    if suffix == ".csv":
        return pd.read_csv(stream)
    if suffix == ".json":
        return pd.read_json(stream)
    if suffix == ".jsonl":
        return pd.read_json(stream, lines=True)
    raise ValueError("Choose a CSV, JSON, or JSONL dataset.")


def extract_log_text(dataset: pd.DataFrame, column: object) -> str:
    """Collect non-empty values from the selected dataset column."""
    if column not in dataset.columns:
        raise ValueError("Choose a column from the uploaded dataset.")
    values = dataset[column].dropna().astype(str).str.strip()
    return "\n".join(values[values != ""])


def discover_patterns(
    log_text: str,
    similarity: float = 0.68,
    min_samples: int = 2,
) -> tuple[int, list[PatternGroup]]:
    """Group similar lines with DBSCAN and return line and pattern counts.

    Identical normalized messages are vectorized once and their frequency is
    passed to DBSCAN as a sample weight. This avoids doing redundant work on
    repeated events while preserving their influence on cluster density.
    """
    if not 0 < similarity < 1:
        raise ValueError("similarity must be greater than 0 and less than 1")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")

    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        return 0, []

    normalized_counts: Counter[str] = Counter()
    example_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for line in lines:
        template = normalize_line(line)
        normalized_counts[template] += 1
        example_counts[template][line] += 1

    templates = list(normalized_counts)
    vectors = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=100_000,
        dtype=np.float32,
    ).fit_transform(templates)
    labels = DBSCAN(
        eps=1 - similarity,
        min_samples=min_samples,
        metric="cosine",
        n_jobs=-1,
    ).fit_predict(vectors, sample_weight=[normalized_counts[item] for item in templates])

    grouped: dict[int | tuple[str, int], list[str]] = defaultdict(list)
    for index, label in enumerate(labels):
        key: int | tuple[str, int] = int(label) if label >= 0 else ("noise", index)
        grouped[key].append(templates[index])

    patterns = []
    for members in grouped.values():
        representative = max(members, key=lambda item: normalized_counts[item])
        example = example_counts[representative].most_common(1)[0][0]
        patterns.append(PatternGroup(
            template=representative,
            example=example,
            count=sum(normalized_counts[item] for item in members),
        ))

    patterns.sort(key=lambda item: (-item.count, item.template.casefold()))
    return len(lines), patterns
