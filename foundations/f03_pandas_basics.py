"""Foundation F03: Pandas operations used in dataset inspection.

Exercises after running this module:
1. Add a ``source`` column and group by both ``split`` and ``source``.
2. Introduce a duplicate row and remove it with ``drop_duplicates``.
3. Export the cleaned frame to JSON Lines and read it back.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd


def parse_jsonl(lines: Iterable[str]) -> pd.DataFrame:
    """Parse in-memory JSON Lines into a DataFrame."""
    records = [json.loads(line) for line in lines if line.strip()]
    return pd.DataFrame.from_records(records)


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a small text dataset without mutating the input."""
    required_columns = {"text", "split"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    cleaned = frame.copy()
    cleaned = cleaned.dropna(subset=["text"])
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned.loc[cleaned["text"].ne("")].copy()

    if "label" not in cleaned:
        cleaned["label"] = "unknown"
    else:
        cleaned["label"] = cleaned["label"].fillna("unknown")

    cleaned["word_count"] = cleaned["text"].str.split().str.len()
    return cleaned.reset_index(drop=True)


def summarize_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sample count and average whitespace-word count by split."""
    return (
        frame.groupby("split", as_index=False)
        .agg(
            examples=("text", "size"),
            average_words=("word_count", "mean"),
        )
        .sort_values("split")
        .reset_index(drop=True)
    )


def attach_source_metadata(
    examples: pd.DataFrame,
    sources: pd.DataFrame,
) -> pd.DataFrame:
    """Join dataset rows with one metadata row per source."""
    return examples.merge(
        sources,
        on="source_id",
        how="left",
        validate="many_to_one",
    )


def run_demo() -> None:
    lines = [
        '{"text":"predict next token","split":"train","label":"lm","source_id":1}',
        '{"text":" inspect padding mask ","split":"train","label":null,"source_id":1}',
        '{"text":"evaluate held out data","split":"validation","label":"eval","source_id":2}',
        '{"text":"","split":"validation","label":"bad","source_id":2}',
    ]
    raw = parse_jsonl(lines)
    cleaned = prepare_dataset(raw)
    summary = summarize_splits(cleaned)
    sources = pd.DataFrame(
        {
            "source_id": [1, 2],
            "source_name": ["synthetic-train", "synthetic-validation"],
        }
    )
    enriched = attach_source_metadata(cleaned, sources)

    assert len(raw) == 4
    assert len(cleaned) == 3
    assert cleaned.loc[1, "label"] == "unknown"
    assert summary["examples"].sum() == 3
    assert enriched["source_name"].notna().all()

    print("raw rows:", len(raw), "clean rows:", len(cleaned))
    print("cleaned columns:", cleaned.columns.tolist())
    print("word_count is whitespace-based, not tokenizer token count")
    print("split summary:\n", summary.to_string(index=False))
    print(
        "joined metadata:\n",
        enriched[["split", "label", "source_name"]].to_string(index=False),
    )


if __name__ == "__main__":
    run_demo()
