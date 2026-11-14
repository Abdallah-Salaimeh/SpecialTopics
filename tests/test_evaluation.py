
import numpy as np
import pandas as pd
import pytest

from specialtopics.evaluation import evaluate_rankings


@pytest.fixture
def evaluation_data():
    return pd.DataFrame({
        "cve_id": ["CVE-1", "CVE-2", "CVE-3", "CVE-4"],
        "published_at": [
            "2025-01-01 08:00:00",
            "2025-01-01 09:00:00",
            "2025-01-01 10:00:00",
            "2025-01-01 11:00:00",
        ],
        "severity": ["CRITICAL", "HIGH", "CRITICAL", "LOW"],
        "assigner_group": ["A", "A", "B", "B"],
        "cwe_ids": ["", "", "", ""],
    })


@pytest.fixture
def evaluation_config():
    return {
        "label": "severity",
        "id": "cve_id",
        "time": "published_at",
        "capacity": {
            "k_per_day": 2,
        },
    }


@pytest.fixture
def evaluation_probabilities():
    return np.array([
        [0.05, 0.05, 0.10, 0.80],
        [0.05, 0.10, 0.75, 0.10],
        [0.05, 0.05, 0.10, 0.80],
        [0.80, 0.10, 0.05, 0.05],
    ])


@pytest.fixture
def evaluation_classes():
    return np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"])


def test_evaluation_returns_expected_metrics(
    evaluation_data,
    evaluation_config,
    evaluation_probabilities,
    evaluation_classes,
):
    results = evaluate_rankings(
        evaluation_data,
        evaluation_probabilities,
        evaluation_classes,
        evaluation_config,
    )

    assert results["model"]["critical_capture_at_k"] == 1.0
    assert results["model"]["hc_capture_at_k"] == pytest.approx(2 / 3)
    assert results["model"]["critical_recall"] == 1.0

    assert results["v0"]["critical_capture_at_k"] == 0.5
    assert results["v0"]["hc_capture_at_k"] == pytest.approx(2 / 3)


def test_group_metrics_use_global_daily_top_k(
    evaluation_data,
    evaluation_config,
    evaluation_probabilities,
    evaluation_classes,
):
    results = evaluate_rankings(
        evaluation_data,
        evaluation_probabilities,
        evaluation_classes,
        evaluation_config,
    )

    assert (
        results["by_assigner_group"]["A"]["model"]["critical_capture_at_k"]
        == 1.0
    )
    assert (
        results["by_assigner_group"]["B"]["model"]["critical_capture_at_k"]
        == 1.0
    )


def test_evaluation_rejects_probability_length_mismatch(
    evaluation_data,
    evaluation_config,
    evaluation_classes,
):
    probabilities = np.array([
        [0.05, 0.05, 0.10, 0.80],
        [0.05, 0.10, 0.75, 0.10],
    ])

    with pytest.raises(ValueError):
        evaluate_rankings(
            evaluation_data,
            probabilities,
            evaluation_classes,
            evaluation_config,
        )


def test_evaluation_rejects_empty_data(
    evaluation_config,
    evaluation_classes,
):
    empty_data = pd.DataFrame(columns=[
        "cve_id",
        "published_at",
        "severity",
        "assigner_group",
        "cwe_ids",
    ])

    probabilities = np.empty((0, 4))

    with pytest.raises(ValueError):
        evaluate_rankings(
            empty_data,
            probabilities,
            evaluation_classes,
            evaluation_config,
        )


def test_evaluation_counts_days_with_k_or_fewer(
    evaluation_config,
    evaluation_classes,
):
    data = pd.DataFrame({
        "cve_id": ["CVE-1", "CVE-2", "CVE-3"],
        "published_at": [
            "2025-01-01 08:00:00",
            "2025-01-01 09:00:00",
            "2025-01-02 08:00:00",
        ],
        "severity": ["CRITICAL", "HIGH", "LOW"],
        "assigner_group": ["A", "A", "B"],
        "cwe_ids": ["", "", ""],
    })

    probabilities = np.array([
        [0.05, 0.05, 0.10, 0.80],
        [0.05, 0.10, 0.75, 0.10],
        [0.80, 0.10, 0.05, 0.05],
    ])

    results = evaluate_rankings(
        data,
        probabilities,
        evaluation_classes,
        evaluation_config,
    )

    assert results["days_with_k_or_fewer"] == 2
