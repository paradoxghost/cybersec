from __future__ import annotations

import logging

import pandas as pd

from src.data.load_data import load_dataset
from src.data.sampling import sample_dataframe
from src.data.split_data import split_dataset
from src.features.preprocess import fit_preprocessor, transform_features
from src.models.metrics import compute_classification_metrics
from src.models.registry import ModelBundle, save_bundle
from src.models.train_baseline import train_baseline_model
from src.models.train_tree_model import train_tree_model
from src.utils.io import load_yaml, save_json
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_seed


def _evaluate(model, prep, le, X: pd.DataFrame, y: pd.Series, benign_label: str):
    Xt = transform_features(prep, X)
    pred_idx = model.predict(Xt)
    pred = le.inverse_transform(pred_idx)
    prob = model.predict_proba(Xt) if hasattr(model, "predict_proba") else None
    return compute_classification_metrics(y.astype(str).to_numpy(), pred, list(le.classes_), benign_label, prob)


def main() -> None:
    configure_logging()
    cfg = load_yaml("configs/config.yaml")
    set_seed(cfg["project"]["random_seed"])
    logger = logging.getLogger("run_training")

    df = load_dataset(cfg["paths"]["raw_data_path"])
    if cfg["sampling"]["enabled"]:
        df = sample_dataframe(
            df,
            cfg["sampling"]["max_rows"],
            cfg["sampling"]["strategy"],
            cfg["schema"]["target_column"],
            cfg["project"]["random_seed"],
        )

    split = split_dataset(
        df=df,
        target_column=cfg["schema"]["target_column"],
        train_size=cfg["split"]["train_size"],
        val_size=cfg["split"]["val_size"],
        test_size=cfg["split"]["test_size"],
        random_seed=cfg["project"]["random_seed"],
        timestamp_column=cfg["schema"]["timestamp_column"],
        group_column=cfg["schema"]["group_column"],
    )

    train_df, val_df, test_df = split["train"], split["val"], split["test"]
    target = cfg["schema"]["target_column"]
    X_train, y_train = train_df.drop(columns=[target]), train_df[target].astype(str)
    X_val, y_val = val_df.drop(columns=[target]), val_df[target].astype(str)
    X_test, y_test = test_df.drop(columns=[target]), test_df[target].astype(str)

    prep_art = fit_preprocessor(
        X_train, y_train, cfg["preprocessing"]["imputer_strategy"], cfg["preprocessing"]["scale_numeric"]
    )

    X_train_t = transform_features(prep_art.feature_pipeline, X_train)
    y_train_enc = prep_art.label_encoder.transform(y_train)

    baseline_model = train_baseline_model(X_train_t, y_train_enc, cfg["models"]["baseline"]["params"])
    tree_model = train_tree_model(X_train_t, y_train_enc, cfg["models"]["stronger"]["params"])

    benign_label = cfg["schema"]["benign_labels"][0]
    baseline_metrics = _evaluate(baseline_model, prep_art.feature_pipeline, prep_art.label_encoder, X_val, y_val, benign_label)
    stronger_metrics = _evaluate(tree_model, prep_art.feature_pipeline, prep_art.label_encoder, X_val, y_val, benign_label)

    save_bundle(ModelBundle("baseline_model", baseline_model, prep_art.feature_pipeline, prep_art.label_encoder, prep_art.feature_columns), cfg["paths"]["models_dir"])
    save_bundle(ModelBundle("stronger_model", tree_model, prep_art.feature_pipeline, prep_art.label_encoder, prep_art.feature_columns), cfg["paths"]["models_dir"])

    train_manifest = {
        "split_strategy": split["split_strategy"],
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "baseline_val_metrics": baseline_metrics,
        "stronger_val_metrics": stronger_metrics,
    }
    save_json(train_manifest, f"{cfg['paths']['reports_metrics_dir']}/training_summary.json")

    test_df.to_parquet(f"{cfg['paths']['processed_dir']}/test_split.parquet", index=False)
    train_df.to_parquet(f"{cfg['paths']['processed_dir']}/train_split.parquet", index=False)
    logger.info("Training complete and artifacts saved.")


if __name__ == "__main__":
    main()
