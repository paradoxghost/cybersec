from __future__ import annotations

import logging

from src.data.load_data import load_dataset
from src.data.sampling import sample_dataframe
from src.data.validate_data import validate_dataset
from src.features.leakage_checks import detect_suspicious_columns
from src.utils.io import load_yaml, save_json
from src.utils.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    cfg = load_yaml("configs/config.yaml")
    logger = logging.getLogger("run_data_validation")

    df = load_dataset(cfg["paths"]["raw_data_path"])
    if cfg["sampling"]["enabled"]:
        df = sample_dataframe(
            df,
            max_rows=cfg["sampling"]["max_rows"],
            strategy=cfg["sampling"]["strategy"],
            target_column=cfg["schema"]["target_column"],
            random_state=cfg["project"]["random_seed"],
        )

    report = validate_dataset(df, target_column=cfg["schema"]["target_column"])
    report["leakage_checks"] = detect_suspicious_columns(
        df, cfg["schema"]["target_column"], cfg["schema"]["id_columns"]
    )

    out = f"{cfg['paths']['reports_metrics_dir']}/data_validation_report.json"
    save_json(report, out)
    logger.info("Validation report saved to %s", out)


if __name__ == "__main__":
    main()
