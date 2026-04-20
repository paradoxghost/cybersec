from __future__ import annotations

import pandas as pd

from src.models.registry import load_bundle
from src.monitoring.alert_rate import compute_alert_rate
from src.monitoring.drift import compute_feature_drift_report
from src.monitoring.reporting import build_monitoring_report
from src.serving.service import InferenceService
from src.utils.io import load_yaml, save_json


def main() -> None:
    cfg = load_yaml("configs/config.yaml")
    target = cfg["schema"]["target_column"]
    benign_labels = cfg["schema"]["benign_labels"]

    train_df = pd.read_parquet(f"{cfg['paths']['processed_dir']}/train_split.parquet")
    test_df = pd.read_parquet(f"{cfg['paths']['processed_dir']}/test_split.parquet")

    bundle = load_bundle(cfg["paths"]["models_dir"], "stronger_model")
    service = InferenceService(bundle, top_k=cfg["serving"]["top_k_probabilities"])

    feature_cols = bundle.feature_columns
    batch_size = cfg["streaming"]["batch_size"]
    outputs = []

    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i : i + batch_size]
        records = batch[feature_cols].to_dict(orient="records")
        preds = service.predict_records(records)
        for pred in preds:
            outputs.append(pred)

    pred_labels = pd.Series([o["predicted_label"] for o in outputs])

    baseline_counts = train_df[target].astype(str).value_counts().to_dict()
    current_counts = pred_labels.value_counts().to_dict()

    baseline_alert_rate = compute_alert_rate(train_df[target].astype(str), benign_labels)
    current_alert_rate = compute_alert_rate(pred_labels, benign_labels)

    feature_drift = compute_feature_drift_report(
        baseline_df=train_df[feature_cols],
        current_df=test_df[feature_cols],
        numeric_columns=feature_cols,
        psi_threshold=cfg["monitoring"]["psi_threshold"],
    )
    pred_alert_report = build_monitoring_report(
        baseline_pred_counts=baseline_counts,
        current_pred_counts=current_counts,
        baseline_alert_rate=baseline_alert_rate,
        current_alert_rate=current_alert_rate,
        jsd_threshold=cfg["monitoring"]["jsd_threshold"],
        alert_spike_threshold=cfg["monitoring"]["alert_rate_spike_threshold"],
    )

    report = {
        "n_batches": (len(test_df) + batch_size - 1) // batch_size,
        "n_predictions": len(outputs),
        "prediction_counts": current_counts,
        "feature_drift": feature_drift,
        "prediction_and_alert_monitoring": pred_alert_report,
    }
    save_json(report, f"{cfg['paths']['reports_metrics_dir']}/stream_monitoring_report.json")
    pd.DataFrame(outputs).to_csv(f"{cfg['paths']['reports_metrics_dir']}/stream_predictions.csv", index=False)


if __name__ == "__main__":
    main()
