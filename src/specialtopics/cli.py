
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_config
from .features import get_feature_columns
from .model import predict_probabilities
from .ranking import rank_predictions


PROJECT_ROOT = Path.cwd()

def validate_batch_input(data, config):
    """Validate the columns required for batch ranking."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The input CSV contains no CVEs.")

    if data.columns.duplicated().any():
        raise ValueError("The input CSV contains duplicate columns.")

    text_columns, categorical_columns, numeric_columns = (
        get_feature_columns(config)
    )

    required_columns = [
        config["id"],
        config["time"],
        "cwe_ids",
        *text_columns,
        *categorical_columns,
        *numeric_columns,
    ]

    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    id_column = config["id"]

    if data[id_column].isna().any():
        raise ValueError("Missing CVE IDs found.")

    if data[id_column].duplicated().any():
        raise ValueError("Duplicate CVE IDs found.")

    return True


def run_batch_ranking(
    input_path,
    output_path,
    config_path=None,
    model_path=None,
    model_version_path=None,
):
    """Load the model, rank CVEs, and export the results."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if config_path is None:
        config_path = PROJECT_ROOT / "features.yaml"

    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "cve-triage.joblib"

    if model_version_path is None:
        model_version_path = PROJECT_ROOT / "models" / "model_version.txt"

    config_path = Path(config_path)
    model_path = Path(model_path)
    model_version_path = Path(model_version_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    if not model_version_path.is_file():
        raise FileNotFoundError(
            f"Model version file not found: {model_version_path}"
        )

    config = load_config(config_path)

    data = pd.read_csv(input_path)

    validate_batch_input(data, config)

    model = joblib.load(model_path)

    probabilities, classes = predict_probabilities(
        model,
        data,
        config
    )

    ranked = rank_predictions(
        data,
        probabilities,
        classes,
        config
    )

    model_version = model_version_path.read_text(
        encoding="utf-8"
    ).strip()

    if not model_version:
        raise ValueError("Model version file is empty.")

    if len(ranked) != len(data):
        raise ValueError(
            "The number of ranked CVEs does not match the input."
        )

    id_column = config["id"]

    output = ranked[
        [id_column, "rank", "score"]
    ].copy()

    output = output.rename(
        columns={id_column: "cve_id"}
    )

    output["model_version"] = model_version

    output = output[
        ["cve_id", "rank", "score", "model_version"]
    ]

    if output["cve_id"].duplicated().any():
        raise ValueError("Duplicate CVEs found in output.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(output_path, index=False)

    print("Batch ranking completed successfully.")
    print(f"Input CVEs: {len(data)}")
    print(f"Output CVEs: {len(output)}")
    print(f"Model version: {model_version}")
    print(f"Output saved to: {output_path}")

    return output


def build_parser():
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Rank CVEs using the trained classification model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CVE CSV file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the ranked CSV will be saved."
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "features.yaml"),
        help="Path to the feature configuration."
    )

    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "cve-triage.joblib"),
        help="Path to the exported trained model."
    )

    parser.add_argument(
        "--model-version",
        default=str(PROJECT_ROOT / "models" / "model_version.txt"),
        help="Path to the model version file."
    )

    return parser


def main(argv=None):
    """Run the batch-ranking command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_batch_ranking(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config,
            model_path=args.model,
            model_version_path=args.model_version,
        )

    except (
        ValueError,
        FileNotFoundError,
        pd.errors.EmptyDataError,
        OSError,
    ) as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    main()
