
import numpy as np
import pandas as pd
import pytest

from specialtopics import cli


@pytest.fixture
def batch_config():
    return {
        "label": "severity",
        "id": "cve_id",
        "time": "published_at",
        "features": {
            "text": ["description"],
            "categorical": [
                "assigner",
                "assigner_group",
                "vendor",
                "product"
            ],
            "numeric": [],
        },
    }


@pytest.fixture
def batch_data():
    return pd.DataFrame({
        "cve_id": [
            "CVE-2026-0001",
            "CVE-2026-0002",
            "CVE-2026-0003",
        ],
        "published_at": [
            "2026-01-02",
            "2026-01-01",
            "2026-01-03",
        ],
        "cwe_ids": [
            "CWE-79",
            "CWE-78",
            "CWE-94",
        ],
        "description": [
            "First vulnerability",
            "Second vulnerability",
            "Third vulnerability",
        ],
        "assigner": ["MITRE", "MITRE", "CISA"],
        "assigner_group": ["Group-A", "Group-A", "Group-B"],
        "vendor": ["Vendor-A", "Vendor-B", "Vendor-C"],
        "product": ["Product-A", "Product-B", "Product-C"],
    })


def test_validate_batch_input(batch_data, batch_config):
    assert cli.validate_batch_input(batch_data, batch_config) is True


def test_validate_batch_input_missing_column(batch_data, batch_config):
    batch_data = batch_data.drop(columns=["description"])

    with pytest.raises(ValueError, match="Missing required columns"):
        cli.validate_batch_input(batch_data, batch_config)


def test_validate_batch_input_empty_data(batch_data, batch_config):
    empty_data = batch_data.iloc[0:0]

    with pytest.raises(ValueError, match="no CVEs"):
        cli.validate_batch_input(empty_data, batch_config)


def test_validate_batch_input_duplicate_ids(batch_data, batch_config):
    batch_data.loc[1, "cve_id"] = batch_data.loc[0, "cve_id"]

    with pytest.raises(ValueError, match="Duplicate CVE IDs"):
        cli.validate_batch_input(batch_data, batch_config)


def test_validate_batch_input_duplicate_columns(batch_config):
    data = pd.DataFrame(
        [["CVE-001", "CVE-002"]],
        columns=["cve_id", "cve_id"]
    )

    with pytest.raises(ValueError, match="duplicate columns"):
        cli.validate_batch_input(data, batch_config)


def test_run_batch_ranking(
    tmp_path,
    batch_data,
    batch_config,
    monkeypatch
):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "ranked.csv"
    config_path = tmp_path / "features.yaml"
    model_path = tmp_path / "cve-triage.joblib"
    version_path = tmp_path / "model_version.txt"

    batch_data.to_csv(input_path, index=False)

    config_path.write_text("test configuration", encoding="utf-8")
    model_path.write_text("mock model", encoding="utf-8")
    version_path.write_text("2\n", encoding="utf-8")

    probabilities = np.array([
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.1, 0.1, 0.8],
        [0.1, 0.1, 0.1, 0.9],
    ])

    classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: batch_config
    )

    monkeypatch.setattr(
        cli.joblib,
        "load",
        lambda path: "mock model"
    )

    monkeypatch.setattr(
        cli,
        "predict_probabilities",
        lambda model, data, config: (probabilities, classes)
    )

    result = cli.run_batch_ranking(
        input_path,
        output_path,
        config_path,
        model_path,
        version_path
    )

    assert output_path.exists()
    assert len(result) == 3

    assert result.columns.tolist() == [
        "cve_id",
        "rank",
        "score",
        "model_version"
    ]

    assert result["cve_id"].tolist() == [
        "CVE-2026-0003",
        "CVE-2026-0002",
        "CVE-2026-0001"
    ]

    assert result["rank"].tolist() == [1, 2, 3]
    assert result["model_version"].tolist() == ["2", "2", "2"]

    saved_result = pd.read_csv(output_path)

    assert len(saved_result) == 3
    assert saved_result["cve_id"].is_unique


def test_run_batch_ranking_missing_model(
    tmp_path,
    batch_data,
    batch_config,
    monkeypatch
):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "ranked.csv"
    model_path = tmp_path / "missing-model.joblib"
    version_path = tmp_path / "model_version.txt"

    config_path = tmp_path / "features.yaml"

    batch_data.to_csv(input_path, index=False)
    version_path.write_text("2", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: batch_config
    )

    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        cli.run_batch_ranking(
            input_path,
            output_path,
            config_path,
            model_path,
            version_path
        )
