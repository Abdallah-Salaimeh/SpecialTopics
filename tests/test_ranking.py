
import numpy as np
import pandas as pd
import pytest

from specialtopics.ranking import (
    has_never_wait_cwe,
    rank_v0,
    rank_predictions,
)


@pytest.fixture
def config():
    return {
        "id": "cve_id",
        "time": "published_at"
    }


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "cve_id": [
            "CVE-001",
            "CVE-002",
            "CVE-003",
            "CVE-004",
            "CVE-005",
            "CVE-006"
        ],
        "published_at": [
            "2026-01-03",
            "2026-01-01",
            "2026-01-02",
            "2026-01-01",
            "2026-01-04",
            "2026-01-02"
        ],
        "cwe_ids": [
            "CWE-79",
            "CWE-78;CWE-79",
            "CWE-787",
            None,
            "CWE-502",
            "CWE-94"
        ]
    })


def test_never_wait_cwe():
    assert has_never_wait_cwe("CWE-78")
    assert has_never_wait_cwe("CWE-94;CWE-79")
    assert not has_never_wait_cwe("CWE-787")
    assert not has_never_wait_cwe("CWE-79")
    assert not has_never_wait_cwe(None)


def test_rank_v0_oldest_first(sample_data, config):
    ranked = rank_v0(sample_data, config)

    assert ranked["cve_id"].tolist() == [
        "CVE-002",
        "CVE-004",
        "CVE-003",
        "CVE-006",
        "CVE-001",
        "CVE-005"
    ]


def test_rank_v0_assigns_ranks(sample_data, config):
    ranked = rank_v0(sample_data, config)

    assert ranked["rank"].tolist() == [1, 2, 3, 4, 5, 6]


def test_rank_predictions_never_wait_first(sample_data, config):
    probabilities = np.tile([0.25, 0.25, 0.25, 0.25], (6, 1))
    classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    ranked = rank_predictions(
        sample_data,
        probabilities,
        classes,
        config
    )

    assert ranked.iloc[0]["never_wait"]
    assert ranked.iloc[1]["never_wait"]
    assert ranked.iloc[2]["never_wait"]
    assert not ranked.iloc[3]["never_wait"]


def test_rank_predictions_probability_order(sample_data, config):
    probabilities = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.1, 0.2, 0.6],
        [0.1, 0.1, 0.2, 0.7],
        [0.1, 0.1, 0.3, 0.5],
        [0.1, 0.1, 0.2, 0.75],
        [0.1, 0.1, 0.2, 0.9]
    ])

    classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    ranked = rank_predictions(
        sample_data,
        probabilities,
        classes,
        config
    )

    assert ranked.iloc[0]["cve_id"] == "CVE-006"
    assert ranked.iloc[1]["cve_id"] == "CVE-005"


def test_rank_predictions_high_probability_tiebreaker(
    sample_data,
    config
):
    data = sample_data.iloc[[0, 2, 3]].copy()

    probabilities = np.array([
        [0.5, 0.1],
        [0.5, 0.8],
        [0.5, 0.3]
    ])

    classes = ["CRITICAL", "HIGH"]

    ranked = rank_predictions(
        data,
        probabilities,
        classes,
        config
    )

    assert ranked["cve_id"].tolist() == [
        "CVE-003",
        "CVE-004",
        "CVE-001"
    ]


def test_rank_predictions_date_and_id_tiebreakers(config):
    data = pd.DataFrame({
        "cve_id": ["CVE-Z", "CVE-B", "CVE-A"],
        "published_at": [
            "2026-01-02",
            "2026-01-01",
            "2026-01-01"
        ],
        "cwe_ids": ["CWE-79", "CWE-79", "CWE-79"]
    })

    probabilities = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])

    classes = ["CRITICAL", "HIGH"]

    ranked = rank_predictions(
        data,
        probabilities,
        classes,
        config
    )

    assert ranked["cve_id"].tolist() == [
        "CVE-A",
        "CVE-B",
        "CVE-Z"
    ]


def test_rank_predictions_length_mismatch(sample_data, config):
    probabilities = np.zeros((5, 4))
    classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    with pytest.raises(ValueError):
        rank_predictions(
            sample_data,
            probabilities,
            classes,
            config
        )


def test_rank_predictions_missing_required_class(sample_data, config):
    probabilities = np.zeros((6, 3))
    classes = ["LOW", "MEDIUM", "CRITICAL"]

    with pytest.raises(ValueError):
        rank_predictions(
            sample_data,
            probabilities,
            classes,
            config
        )
