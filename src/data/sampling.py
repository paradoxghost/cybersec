"""Sampling helpers for large datasets."""
from __future__ import annotations

import pandas as pd


def _validate_target_present(sampled_df: pd.DataFrame, target_column: str) -> None:
    if target_column not in sampled_df.columns:
        raise ValueError(
            f"Sampling produced a DataFrame without target column '{target_column}'. "
            "Verify sampling strategy and input schema."
        )


def sample_dataframe(
    df: pd.DataFrame,
    max_rows: int,
    strategy: str,
    target_column: str,
    random_state: int,
) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        sampled_df = df.copy()
        if target_column in df.columns:
            _validate_target_present(sampled_df, target_column)
        return sampled_df

    if strategy == "stratified" and target_column in df.columns:
        frac = max_rows / len(df)
        sampled_df = (
            df.groupby(target_column, group_keys=False, as_index=False)
            .apply(lambda x: x.sample(max(1, int(len(x) * frac)), random_state=random_state))
            .reset_index(drop=True)
        )
        if len(sampled_df) > max_rows:
            sampled_df = sampled_df.sample(max_rows, random_state=random_state).reset_index(drop=True)
    else:
        sampled_df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)

    if target_column in df.columns:
        _validate_target_present(sampled_df, target_column)

    return sampled_df
