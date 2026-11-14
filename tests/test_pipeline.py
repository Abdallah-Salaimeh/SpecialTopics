
import json
from contextlib import nullcontext

import pandas as pd
import yaml

from specialtopics import pipeline


def test_pipeline_runs_end_to_end_and_writes_reports(
    tmp_path,
    monkeypatch,
):
    dates = pd.date_range(
        "2025-01-01",
        "2025-06-30",
        periods=200,
    )

    data = pd.DataFrame({
        "cve_id": [f"CVE-2025-{i:05d}" for i in range(200)],
        "published_at": dates.astype(str),
        "description": [
            f"Unique vulnerability description number {i}"
            for i in range(200)
        ],
        "assigner": [f"assigner-{i % 4}" for i in range(200)],
        "assigner_group": [
            "top10" if i % 2 == 0 else "other"
            for i in range(200)
        ],
        "vendor": [f"vendor-{i % 5}" for i in range(200)],
        "product": [f"product-{i % 7}" for i in range(200)],
        "cwe_ids": [""] * 200,
        "severity": [
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"][i % 4]
            for i in range(200)
        ],
    })

    data_path = tmp_path / "cves_history.csv"
    data.to_csv(data_path, index=False)

    features_config = {
        "label": "severity",
        "id": "cve_id",
        "time": "published_at",
        "features": {
            "text": ["description"],
            "categorical": [
                "assigner",
                "assigner_group",
                "vendor",
                "product",
            ],
            "numeric": [],
        },
        "not_features": {
            "cwe_ids": "Used only by the never-wait ranking rule."
        },
        "groups": ["assigner_group"],
    }

    pipeline_config = {
        "capacity": {"k_per_day": 5},
        "targets": {
            "critical_capture_at_k": {
                "overall": 0.0,
                "min_per_assigner_group": 0.0,
            },
            "hc_capture_at_k": {"overall": 0.0},
            "beat_v0_by": 0.0,
        },
    }

    features_path = tmp_path / "features.yaml"
    config_path = tmp_path / "config.yaml"

    features_path.write_text(
        yaml.safe_dump(features_config),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(pipeline_config),
        encoding="utf-8",
    )

    reports_path = tmp_path / "reports"

    # Mock MLflow tracking so this test does not create a real run.
    monkeypatch.setattr(
        pipeline.mlflow,
        "set_tracking_uri",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "set_experiment",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "start_run",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_metrics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline.mlflow.sklearn,
        "log_model",
        lambda *args, **kwargs: None,
    )

    results = pipeline.run_pipeline(
        data_path=data_path,
        features_path=features_path,
        config_path=config_path,
        reports_path=reports_path,
    )

    metrics_path = reports_path / "metrics.json"
    report_path = reports_path / "eval_report.md"

    assert metrics_path.is_file()
    assert report_path.is_file()

    with metrics_path.open("r", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert results["days_evaluated"] > 0
    assert results["duplicates_removed"] == 0
    assert "selected_model_params" in results
    assert len(results["validation_candidates"]) == 6
    assert saved_metrics["days_evaluated"] > 0

    report_text = report_path.read_text(encoding="utf-8")
    assert "Critical Capture@K" in report_text
