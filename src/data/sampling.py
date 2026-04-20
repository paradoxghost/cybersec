"""Sampling helpers for large datasets."""
from __future__ import annotations

import pandas as pd


def sample_dataframe(
    df: pd.DataFrame,
    max_rows: int,
    strategy: str,
    target_column: str,
    random_state: int,
) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    if strategy == "stratified" and target_column in df.columns:
        frac = max_rows / len(df)
        sampled = (
            df.groupby(target_column, group_keys=False)
            .apply(lambda x: x.sample(max(1, int(len(x) * frac)), random_state=random_state))
            .reset_index(drop=True)
        )
        if len(sampled) > max_rows:
            sampled = sampled.sample(max_rows, random_state=random_state).reset_index(drop=True)
        return sampled
    return df.sample(max_rows, random_state=random_state).reset_index(drop=True)
