
import numpy as np
import pandas as pd
import pytest #type: ignore

from specialtopics.model import build_model, train_model, predict_severity, predict_probabilities, predict_critical_probability #type: ignore


@pytest.fixture
def config():
    return {
        "label": "severity",
        "features": {
            "text": ["description"],
            "categorical": ["assigner", "assigner_group", "vendor", "product"],
            "numeric": []
        }
    }


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "description": ["low vulnerability"] * 5 + ["medium vulnerability"] * 5 + ["high vulnerability"] * 5 + ["critical vulnerability"] * 5,
        "assigner": ["MITRE", "CNA1", "CNA2", "MITRE", "CNA1"] * 4,
        "assigner_group": ["GroupA", "GroupB", "GroupA", "GroupB", "GroupA"] * 4,
        "vendor": ["VendorA", "VendorB", "VendorC", "VendorA", "VendorB"] * 4,
        "product": ["ProductA", "ProductB", "ProductC", "ProductA", "ProductB"] * 4,
        "severity": ["LOW"] * 5 + ["MEDIUM"] * 5 + ["HIGH"] * 5 + ["CRITICAL"] * 5
    })


@pytest.fixture
def trained_model(sample_data, config):
    return train_model(sample_data, config), config


def test_build_model(config):
    model = build_model(config)
    assert model is not None
    assert len(model.steps) == 2


def test_train_model(sample_data, config):
    model = train_model(sample_data, config)
    assert hasattr(model, "classes_")


def test_predict_severity(trained_model, sample_data):
    model, config = trained_model
    predictions = predict_severity(model, sample_data.drop(columns=["severity"]), config)
    assert len(predictions) == len(sample_data)
    assert set(predictions).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def test_predict_probabilities(trained_model, sample_data):
    model, config = trained_model
    probabilities, classes = predict_probabilities(model, sample_data.drop(columns=["severity"]), config)
    assert probabilities.shape == (len(sample_data), len(classes))


def test_probability_range(trained_model, sample_data):
    model, config = trained_model
    probabilities, _ = predict_probabilities(model, sample_data.drop(columns=["severity"]), config)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_probability_sum(trained_model, sample_data):
    model, config = trained_model
    probabilities, _ = predict_probabilities(model, sample_data.drop(columns=["severity"]), config)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_unknown_category(trained_model, sample_data):
    model, config = trained_model
    new_data = sample_data.drop(columns=["severity"]).iloc[[0]].copy()
    new_data["vendor"] = "UnknownVendor"
    predictions = predict_severity(model, new_data, config)
    assert len(predictions) == 1


def test_missing_description(trained_model, sample_data):
    model, config = trained_model
    new_data = sample_data.drop(columns=["severity"]).iloc[[0]].copy()
    new_data["description"] = None
    predictions = predict_severity(model, new_data, config)
    assert len(predictions) == 1


def test_critical_probability(trained_model, sample_data):
    model, config = trained_model
    probabilities = predict_critical_probability(model, sample_data.drop(columns=["severity"]), config)
    assert len(probabilities) == len(sample_data)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_missing_critical_class(config):
    data = pd.DataFrame({
        "description": ["low vulnerability", "medium vulnerability", "high vulnerability", "low issue", "medium issue", "high issue"],
        "assigner": ["A", "B", "C", "A", "B", "C"],
        "assigner_group": ["G1", "G2", "G3", "G1", "G2", "G3"],
        "vendor": ["V1", "V2", "V3", "V1", "V2", "V3"],
        "product": ["P1", "P2", "P3", "P1", "P2", "P3"],
        "severity": ["LOW", "MEDIUM", "HIGH", "LOW", "MEDIUM", "HIGH"]
    })

    model = train_model(data, config)

    with pytest.raises(ValueError, match="CRITICAL"):
        predict_critical_probability(model, data.drop(columns=["severity"]), config)
