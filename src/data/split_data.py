"""Leakage-aware data split logic."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int,
    timestamp_column: str | None = None,
    group_column: str | None = None,
) -> dict[str, Any]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    y = df[target_column]

    if timestamp_column and timestamp_column in df.columns:
        sorted_df = df.sort_values(timestamp_column).reset_index(drop=True)
        n = len(sorted_df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        return {
            "train": sorted_df.iloc[:train_end].copy(),
            "val": sorted_df.iloc[train_end:val_end].copy(),
            "test": sorted_df.iloc[val_end:].copy(),
            "split_strategy": "time_aware",
        }

    if group_column and group_column in df.columns:
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_seed)
        train_idx, hold_idx = next(gss.split(df, y=y, groups=df[group_column]))
        train_df = df.iloc[train_idx].copy()
        hold_df = df.iloc[hold_idx].copy()

        hold_ratio = test_size / (test_size + val_size)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=(1 - hold_ratio), random_state=random_seed)
        val_idx, test_idx = next(gss2.split(hold_df, groups=hold_df[group_column]))
        return {
            "train": train_df,
            "val": hold_df.iloc[val_idx].copy(),
            "test": hold_df.iloc[test_idx].copy(),
            "split_strategy": "group_aware",
        }

    train_df, hold_df = train_test_split(
        df,
        train_size=train_size,
        stratify=y,
        random_state=random_seed,
    )
    hold_test_ratio = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        hold_df,
        test_size=hold_test_ratio,
        stratify=hold_df[target_column],
        random_state=random_seed,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
        "split_strategy": "stratified_random",
    }
