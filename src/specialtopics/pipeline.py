
import argparse
import re
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from .data import load_config, load_dataset, validate_dataset
from .model import train_model, predict_probabilities
from .evaluation import evaluate_rankings
from .reporting import save_metrics, generate_markdown_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_CANDIDATES = [
    {"C": 0.1, "class_weight": None},
    {"C": 1.0, "class_weight": None},
    {"C": 10.0, "class_weight": None},
    {"C": 0.1, "class_weight": "balanced"},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 10.0, "class_weight": "balanced"},
]


def normalize_description(description):
    if pd.isna(description):
        return ""

    description = str(description).lower()
    description = re.sub(
        r"\bcve-\d{4}-\d+\b",
        " ",
        description,
        flags=re.IGNORECASE,
    )
    description = re.sub(r"\s+", " ", description).strip()

    return description


def remove_test_duplicates(train_data, test_data):
    train_descriptions = {
        normalize_description(description)
        for description in train_data["description"]
        if normalize_description(description)
    }

    normalized_test = test_data["description"].apply(
        normalize_description
    )

    duplicate_mask = normalized_test.isin(train_descriptions)

    return (
        test_data.loc[~duplicate_mask].copy(),
        int(duplicate_mask.sum()),
    )


def split_dataset(data, time_column):
    dates = pd.to_datetime(
        data[time_column],
        errors="coerce",
        utc=True,
    )

    if dates.isna().any():
        raise ValueError(
            "Invalid publication dates found during splitting."
        )

    months = dates.dt.tz_localize(None).dt.to_period("M")
    latest_month = months.max()

    test_start = latest_month - 1
    validation_start = latest_month - 3

    train_mask = months < validation_start
    validation_mask = (
        (months >= validation_start)
        & (months < test_start)
    )
    test_mask = months >= test_start

    train_data = data.loc[train_mask].copy()
    validation_data = data.loc[validation_mask].copy()
    test_data = data.loc[test_mask].copy()

    if (
        train_data.empty
        or validation_data.empty
        or test_data.empty
    ):
        raise ValueError(
            "Insufficient historical data for the required "
            "chronological split."
        )

    periods = {
        "training": f"Before {validation_start}",
        "validation": (
            f"{validation_start} to {test_start - 1}"
        ),
        "test": f"{test_start} to {latest_month}",
    }

    return train_data, validation_data, test_data, periods


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def calculate_target_results(metrics, config):
    targets = config.get("targets", {})
    results = {}

    critical_target = targets.get(
        "critical_capture_at_k",
        {},
    )
    actual_critical = metrics["model"]["critical_capture_at_k"]

    if actual_critical is not None and "overall" in critical_target:
        required = critical_target["overall"]
        results["critical_capture_at_k_overall"] = {
            "required": required,
            "actual": actual_critical,
            "met": actual_critical >= required,
        }

    hc_target = targets.get("hc_capture_at_k", {})
    actual_hc = metrics["model"]["hc_capture_at_k"]

    if actual_hc is not None and "overall" in hc_target:
        required = hc_target["overall"]
        results["hc_capture_at_k_overall"] = {
            "required": required,
            "actual": actual_hc,
            "met": actual_hc >= required,
        }

    group_target = critical_target.get(
        "min_per_assigner_group"
    )

    if group_target is not None:
        group_values = [
            values["model"]["critical_capture_at_k"]
            for values in metrics["by_assigner_group"].values()
            if values["model"]["critical_capture_at_k"] is not None
        ]

        if group_values:
            actual = min(group_values)
            results["critical_capture_at_k_min_group"] = {
                "required": group_target,
                "actual": actual,
                "met": actual >= group_target,
            }

    improvement_target = targets.get("beat_v0_by")
    model_critical = metrics["model"]["critical_capture_at_k"]
    v0_critical = metrics["v0"]["critical_capture_at_k"]

    if (
        improvement_target is not None
        and model_critical is not None
        and v0_critical is not None
    ):
        improvement = model_critical - v0_critical

        results["improvement_over_v0"] = {
            "required": improvement_target,
            "actual": improvement,
            "met": improvement >= improvement_target,
        }

    return results


def log_metrics_to_mlflow(metrics, prefix):
    values = {}

    for name, value in metrics["model"].items():
        if isinstance(value, (int, float)) and value is not None:
            values[f"{prefix}_model_{name}"] = value

    for name, value in metrics["v0"].items():
        if isinstance(value, (int, float)) and value is not None:
            values[f"{prefix}_v0_{name}"] = value

    if values:
        mlflow.log_metrics(values)


def select_model_on_validation(
    train_data,
    validation_data,
    feature_config,
    evaluation_config,
    features_path,
    commit,
):
    """
    Compare candidate configurations using validation data only.

    Selection priority:
    1. Critical Capture@K
    2. HC Capture@K
    3. Critical recall

    The test set is never used during model selection.
    """
    best_params = None
    best_key = None
    candidate_results = []

    for candidate_number, candidate_params in enumerate(
        MODEL_CANDIDATES,
        start=1,
    ):
        with mlflow.start_run(
            run_name=f"validation-candidate-{candidate_number}"
        ):
            model = train_model(
                train_data,
                feature_config,
                candidate_params,
            )

            probabilities, classes = predict_probabilities(
                model,
                validation_data,
                feature_config,
            )

            metrics = evaluate_rankings(
                validation_data,
                probabilities,
                classes,
                evaluation_config,
            )

            critical_capture = (
                metrics["model"]["critical_capture_at_k"]
            )
            hc_capture = metrics["model"]["hc_capture_at_k"]
            critical_recall = metrics["model"]["critical_recall"]

            selection_key = (
                critical_capture
                if critical_capture is not None
                else -1,
                hc_capture if hc_capture is not None else -1,
                critical_recall if critical_recall is not None else -1,
            )

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("candidate_number", candidate_number)
            mlflow.log_param("C", candidate_params["C"])
            mlflow.log_param(
                "class_weight",
                str(candidate_params["class_weight"]),
            )
            mlflow.log_param("training_rows", len(train_data))
            mlflow.log_param("validation_rows", len(validation_data))
            mlflow.log_param("git_commit", commit)
            mlflow.log_param(
                "k_per_day",
                evaluation_config["capacity"]["k_per_day"],
            )

            log_metrics_to_mlflow(metrics, "validation")
            mlflow.log_artifact(str(features_path))

            candidate_results.append({
                "params": candidate_params.copy(),
                "critical_capture_at_k": critical_capture,
                "hc_capture_at_k": hc_capture,
                "critical_recall": critical_recall,
            })

            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_params = candidate_params.copy()

    if best_params is None:
        raise RuntimeError(
            "Model selection failed: no candidate was evaluated."
        )

    return best_params, candidate_results


def run_pipeline(
    data_path=None,
    features_path=None,
    config_path=None,
    reports_path=None,
):
    data_path = Path(
        data_path or PROJECT_ROOT / "data" / "cves_history.csv"
    )
    features_path = Path(
        features_path or PROJECT_ROOT / "features.yaml"
    )
    config_path = Path(
        config_path or PROJECT_ROOT / "config.yaml"
    )
    reports_path = Path(
        reports_path or PROJECT_ROOT / "reports"
    )

    feature_config = load_config(features_path)
    pipeline_config = load_config(config_path)

    evaluation_config = {
        **feature_config,
        **pipeline_config,
    }

    data = load_dataset(data_path)
    validate_dataset(data, feature_config)

    time_column = feature_config["time"]
    label_column = feature_config["label"]

    train_data, validation_data, test_data, periods = (
        split_dataset(data, time_column)
    )

    test_data, duplicates_removed = remove_test_duplicates(
        train_data,
        test_data,
    )

    if test_data.empty:
        raise ValueError(
            "No test CVEs remain after duplicate removal."
        )

    if test_data[label_column].nunique() < 1:
        raise ValueError(
            "Test dataset has no severity labels."
        )

    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("CVE Triage Assistant")

    commit = get_git_commit()

    best_params, candidate_results = select_model_on_validation(
        train_data,
        validation_data,
        feature_config,
        evaluation_config,
        features_path,
        commit,
    )

    print("Validation model selection completed.")
    print(f"Selected parameters: {best_params}")

    final_training_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    with mlflow.start_run(run_name="final-test-model"):
        final_model = train_model(
            final_training_data,
            feature_config,
            best_params,
        )

        test_probabilities, test_classes = predict_probabilities(
            final_model,
            test_data,
            feature_config,
        )

        test_metrics = evaluate_rankings(
            test_data,
            test_probabilities,
            test_classes,
            evaluation_config,
        )

        test_metrics["duplicates_removed"] = duplicates_removed
        test_metrics["periods"] = periods
        test_metrics["selected_model_params"] = best_params
        test_metrics["validation_candidates"] = candidate_results

        test_metrics["target_results"] = calculate_target_results(
            test_metrics,
            pipeline_config,
        )

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("selected_C", best_params["C"])
        mlflow.log_param(
            "selected_class_weight",
            str(best_params["class_weight"]),
        )
        mlflow.log_param("training_rows", len(final_training_data))
        mlflow.log_param("test_rows", len(test_data))
        mlflow.log_param("duplicates_removed", duplicates_removed)
        mlflow.log_param("git_commit", commit)
        mlflow.log_param(
            "k_per_day",
            pipeline_config["capacity"]["k_per_day"],
        )

        log_metrics_to_mlflow(test_metrics, "test")

        mlflow.log_artifact(str(features_path))
        mlflow.log_artifact(str(config_path))

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            registered_model_name="cve-triage",
            skops_trusted_types=["numpy.dtype"],
        )

    save_metrics(
        test_metrics,
        reports_path / "metrics.json",
    )

    generate_markdown_report(
        test_metrics,
        reports_path / "eval_report.md",
    )

    print("Pipeline completed successfully.")
    print(f"Training rows: {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Test duplicates removed: {duplicates_removed}")
    print(f"Selected model parameters: {best_params}")
    print(f"Reports saved to: {reports_path}")
    print(f"MLflow tracking URI: {tracking_uri}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="CVE Triage Training Pipeline"
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Historical CSV path",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Feature YAML path",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline YAML path",
    )
    parser.add_argument(
        "--reports",
        default=None,
        help="Reports output directory",
    )

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        features_path=args.features,
        config_path=args.config,
        reports_path=args.reports,
    )


if __name__ == "__main__":
    main()
