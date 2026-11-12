
from pathlib import Path

import pandas as pd
import yaml


VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def load_config(config_path):
    """Load the feature configuration from YAML."""

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Feature configuration must be a YAML mapping."
        )

    return config


def load_dataset(csv_path):
    """Load a CVE dataset from a CSV file."""

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}"
        )

    return pd.read_csv(csv_path)


def validate_dataset(df, config):
    """Validate the CVE dataset against the feature configuration."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns found.")

    features = config.get("features", {})
    expected_columns = []

    # Collect model feature columns.
    for role_columns in features.values():
        if isinstance(role_columns, list):
            expected_columns.extend(role_columns)

    # Collect label, ID, and time columns.
    for key in ["label", "id", "time"]:
        if key in config:
            expected_columns.append(config[key])

    # Collect columns excluded from model features.
    not_features = config.get("not_features", {})

    if isinstance(not_features, dict):
        expected_columns.extend(not_features.keys())

    elif isinstance(not_features, list):
        expected_columns.extend(not_features)

    # Remove duplicates.
    expected_columns = list(dict.fromkeys(expected_columns))

    actual_columns = list(df.columns)

    # Check missing and unexpected columns.
    missing_columns = set(expected_columns) - set(actual_columns)
    unexpected_columns = set(actual_columns) - set(expected_columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns: {unexpected_columns}"
        )

    # Validate label.
    label_column = config.get("label")

    if not label_column or label_column not in df.columns:
        raise ValueError("Label column is missing.")

    if df[label_column].isna().any():
        raise ValueError("Missing labels found.")

    invalid_labels = (
        set(df[label_column].dropna().unique()) - VALID_SEVERITIES
    )

    if invalid_labels:
        raise ValueError(
            f"Unsupported labels: {invalid_labels}"
        )

    # Validate CVE IDs.
    id_column = config.get("id", "cve_id")

    if id_column not in df.columns:
        raise ValueError("ID column is missing.")

    if df[id_column].isna().any():
        raise ValueError("Missing CVE IDs found.")

    if df[id_column].duplicated().any():
        raise ValueError("Duplicate CVE IDs found.")

    return True
