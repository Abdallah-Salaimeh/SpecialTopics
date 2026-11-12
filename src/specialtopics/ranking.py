
import pandas as pd


NEVER_WAIT_CWES = {
    "CWE-78",
    "CWE-94",
    "CWE-287",
    "CWE-306",
    "CWE-502"
}


def has_never_wait_cwe(value):
    """Check whether a CVE contains one of the never-wait CWEs."""

    if pd.isna(value):
        return False

    cwe_values = {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }

    return bool(cwe_values & NEVER_WAIT_CWES)


def rank_v0(data, config):
    """Rank CVEs using the oldest publication date first."""

    id_column = config["id"]
    time_column = config["time"]

    ranked = data.copy()

    ranked[time_column] = pd.to_datetime(
        ranked[time_column],
        errors="coerce",
        utc=True
    )

    ranked = ranked.sort_values(
        [time_column, id_column],
        ascending=[True, True],
        na_position="last"
    ).reset_index(drop=True)

    ranked["rank"] = range(1, len(ranked) + 1)

    return ranked


def rank_predictions(data, probabilities, classes, config):
    """Rank CVEs using never-wait priority and model probabilities."""

    id_column = config["id"]
    time_column = config["time"]

    if len(data) != len(probabilities):
        raise ValueError(
            "The number of predictions must match the number of CVEs."
        )

    classes = list(classes)

    if "CRITICAL" not in classes or "HIGH" not in classes:
        raise ValueError(
            "The model must provide CRITICAL and HIGH probabilities."
        )

    critical_index = classes.index("CRITICAL")
    high_index = classes.index("HIGH")

    ranked = data.copy()

    ranked["score"] = probabilities[:, critical_index]
    ranked["high_probability"] = probabilities[:, high_index]

    ranked["never_wait"] = ranked["cwe_ids"].apply(
        has_never_wait_cwe
    )

    ranked[time_column] = pd.to_datetime(
        ranked[time_column],
        errors="coerce",
        utc=True
    )

    ranked = ranked.sort_values(
        [
            "never_wait",
            "score",
            "high_probability",
            time_column,
            id_column
        ],
        ascending=[False, False, False, True, True],
        na_position="last"
    ).reset_index(drop=True)

    ranked["rank"] = range(1, len(ranked) + 1)

    return ranked
