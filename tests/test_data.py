import pandas as pd
import pytest

from specialtopics.data import (
    load_config,
    load_dataset,
    validate_dataset,
)

CONFIG_PATH = "features.yaml"
SAMPLE_PATH = "tests/data/sample_cves.csv"


def test_load_config():
    config = load_config(CONFIG_PATH)
    assert config is not None


def test_load_dataset():
    df = load_dataset(SAMPLE_PATH)
    assert len(df) == 200


def test_valid_dataset():
    config = load_config(CONFIG_PATH)
    df = load_dataset(SAMPLE_PATH)
    assert validate_dataset(df, config) is True


def test_missing_column():
    config = load_config(CONFIG_PATH)
    df = load_dataset(SAMPLE_PATH)
    df = df.drop(columns=["description"])
    with pytest.raises(ValueError):
        validate_dataset(df, config)


def test_unexpected_column():
    config = load_config(CONFIG_PATH)
    df = load_dataset(SAMPLE_PATH)
    df["unexpected_column"] = "test"
    with pytest.raises(ValueError):
        validate_dataset(df, config)


def test_duplicate_cve_id():
    config = load_config(CONFIG_PATH)
    df = load_dataset(SAMPLE_PATH)
    df.loc[1, "cve_id"] = df.loc[0, "cve_id"]
    with pytest.raises(ValueError):
        validate_dataset(df, config)


def test_invalid_label():
    config = load_config(CONFIG_PATH)
    df = load_dataset(SAMPLE_PATH)
    df.loc[0, "severity"] = "INVALID"
    with pytest.raises(ValueError):
        validate_dataset(df, config)