
import numpy as np
import pandas as pd

from .ranking import rank_v0, rank_predictions


def evaluate_rankings(data, probabilities, classes, config):
    label_column = config["label"]
    time_column = config["time"]
    id_column = config["id"]
    k = config["capacity"]["k_per_day"]

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("Capacity K must be a positive integer.")

    required_columns = [
        label_column,
        time_column,
        id_column,
        "assigner_group",
        "cwe_ids"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Cannot evaluate an empty dataset.")

    if data[label_column].isna().any():
        raise ValueError("Missing labels found during evaluation.")

    if data[id_column].isna().any():
        raise ValueError("Missing CVE IDs found during evaluation.")

    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)

    if probabilities.ndim != 2:
        raise ValueError("Prediction probabilities must be two-dimensional.")

    if len(data) != len(probabilities):
        raise ValueError("Data and prediction probabilities have different lengths.")

    if probabilities.shape[1] != len(classes):
        raise ValueError("Probability columns do not match the supplied classes.")

    if len(set(classes)) != len(classes):
        raise ValueError("Duplicate prediction classes found.")

    if not {"HIGH", "CRITICAL"}.issubset(set(classes)):
        raise ValueError("Model probabilities must contain HIGH and CRITICAL classes.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities contain invalid values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Prediction probabilities must be between 0 and 1.")

    evaluated_data = data.copy().reset_index(drop=True)
    evaluated_data["_row_position"] = np.arange(len(evaluated_data))

    evaluated_data["_evaluation_day"] = pd.to_datetime(
        evaluated_data[time_column],
        errors="coerce",
        utc=True
    ).dt.date

    if evaluated_data["_evaluation_day"].isna().any():
        raise ValueError("Invalid publication dates found during evaluation.")

    actual_labels = evaluated_data[label_column].to_numpy()
    predicted_labels = classes[np.argmax(probabilities, axis=1)]

    critical_total = int(np.sum(actual_labels == "CRITICAL"))
    hc_total = int(np.isin(actual_labels, ["HIGH", "CRITICAL"]).sum())

    critical_true_positives = int(
        np.sum(
            (actual_labels == "CRITICAL") &
            (predicted_labels == "CRITICAL")
        )
    )

    critical_recall = (
        critical_true_positives / critical_total
        if critical_total else None
    )

    critical_captured = 0
    hc_captured = 0
    v0_critical_captured = 0
    v0_hc_captured = 0

    group_counts = {}
    daily_results = []
    days_with_k_or_fewer = 0

    for day, day_data in evaluated_data.groupby("_evaluation_day", sort=True):
        positions = day_data["_row_position"].to_numpy()
        day_probabilities = probabilities[positions]

        model_ranked = rank_predictions(
            day_data,
            day_probabilities,
            classes,
            config
        )

        v0_ranked = rank_v0(day_data, config)

        model_top_k = model_ranked.head(k)
        v0_top_k = v0_ranked.head(k)

        critical_total_day = int(
            day_data[label_column].eq("CRITICAL").sum()
        )

        hc_total_day = int(
            day_data[label_column].isin(["HIGH", "CRITICAL"]).sum()
        )

        model_critical = int(
            model_top_k[label_column].eq("CRITICAL").sum()
        )

        model_hc = int(
            model_top_k[label_column].isin(["HIGH", "CRITICAL"]).sum()
        )

        v0_critical = int(
            v0_top_k[label_column].eq("CRITICAL").sum()
        )

        v0_hc = int(
            v0_top_k[label_column].isin(["HIGH", "CRITICAL"]).sum()
        )

        critical_captured += model_critical
        hc_captured += model_hc
        v0_critical_captured += v0_critical
        v0_hc_captured += v0_hc

        if len(day_data) <= k:
            days_with_k_or_fewer += 1

        for group, group_data in day_data.groupby(
            "assigner_group",
            dropna=False
        ):
            group_name = "UNKNOWN" if pd.isna(group) else str(group)

            if group_name not in group_counts:
                group_counts[group_name] = {
                    "model_critical_captured": 0,
                    "model_critical_total": 0,
                    "model_hc_captured": 0,
                    "model_hc_total": 0,
                    "v0_critical_captured": 0,
                    "v0_hc_captured": 0
                }

            group_counts[group_name]["model_critical_total"] += int(
                group_data[label_column].eq("CRITICAL").sum()
            )

            group_counts[group_name]["model_hc_total"] += int(
                group_data[label_column].isin(["HIGH", "CRITICAL"]).sum()
            )

        for _, row in model_top_k.iterrows():
            group = row["assigner_group"]
            group_name = "UNKNOWN" if pd.isna(group) else str(group)

            if row[label_column] == "CRITICAL":
                group_counts[group_name]["model_critical_captured"] += 1

            if row[label_column] in ["HIGH", "CRITICAL"]:
                group_counts[group_name]["model_hc_captured"] += 1

        for _, row in v0_top_k.iterrows():
            group = row["assigner_group"]
            group_name = "UNKNOWN" if pd.isna(group) else str(group)

            if row[label_column] == "CRITICAL":
                group_counts[group_name]["v0_critical_captured"] += 1

            if row[label_column] in ["HIGH", "CRITICAL"]:
                group_counts[group_name]["v0_hc_captured"] += 1

        daily_results.append({
            "day": str(day),
            "cve_count": len(day_data),
            "critical_total": critical_total_day,
            "critical_captured": model_critical,
            "hc_total": hc_total_day,
            "hc_captured": model_hc,
            "v0_critical_captured": v0_critical,
            "v0_hc_captured": v0_hc
        })

    group_metrics = {}

    for group, values in group_counts.items():
        group_metrics[group] = {
            "model": {
                "critical_capture_at_k": (
                    values["model_critical_captured"] /
                    values["model_critical_total"]
                    if values["model_critical_total"] else None
                ),
                "hc_capture_at_k": (
                    values["model_hc_captured"] /
                    values["model_hc_total"]
                    if values["model_hc_total"] else None
                )
            },
            "v0": {
                "critical_capture_at_k": (
                    values["v0_critical_captured"] /
                    values["model_critical_total"]
                    if values["model_critical_total"] else None
                ),
                "hc_capture_at_k": (
                    values["v0_hc_captured"] /
                    values["model_hc_total"]
                    if values["model_hc_total"] else None
                )
            },
            **values
        }

    return {
        "k_per_day": k,
        "days_evaluated": len(daily_results),
        "days_with_k_or_fewer": days_with_k_or_fewer,
        "model": {
            "critical_capture_at_k": (
                critical_captured / critical_total
                if critical_total else None
            ),
            "hc_capture_at_k": (
                hc_captured / hc_total
                if hc_total else None
            ),
            "critical_captured": critical_captured,
            "critical_total": critical_total,
            "hc_captured": hc_captured,
            "hc_total": hc_total,
            "critical_recall": critical_recall
        },
        "v0": {
            "critical_capture_at_k": (
                v0_critical_captured / critical_total
                if critical_total else None
            ),
            "hc_capture_at_k": (
                v0_hc_captured / hc_total
                if hc_total else None
            ),
            "critical_captured": v0_critical_captured,
            "hc_captured": v0_hc_captured
        },
        "by_assigner_group": group_metrics,
        "daily_results": daily_results
    }
