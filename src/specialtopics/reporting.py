
import json
from pathlib import Path


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, allow_nan=False)


def generate_markdown_report(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = metrics["model"]
    v0 = metrics["v0"]

    lines = [
        "# CVE Triage Evaluation Report",
        "",
        "## Evaluation Overview",
        "",
        f"- Capacity per day: {metrics['k_per_day']}",
        f"- Days evaluated: {metrics['days_evaluated']}",
        f"- Days with K or fewer CVEs: {metrics['days_with_k_or_fewer']}",
        f"- Test duplicates removed: {metrics['duplicates_removed']}",
        f"- Training period: {metrics['periods']['training']}",
        f"- Validation period: {metrics['periods']['validation']}",
        f"- Test period: {metrics['periods']['test']}",
        "",
        "## Overall Results",
        "",
        "| Metric | ML Model | V0 Baseline |",
        "|---|---:|---:|",
        f"| Critical Capture@K | {model['critical_capture_at_k']} | {v0['critical_capture_at_k']} |",
        f"| HC Capture@K | {model['hc_capture_at_k']} | {v0['hc_capture_at_k']} |",
        f"| Critical Recall | {model['critical_recall']} | N/A |",
        "",
        "## Results by Assigner Group",
        "",
        "| Group | Model Critical Capture@K | V0 Critical Capture@K | Model HC Capture@K | V0 HC Capture@K |",
        "|---|---:|---:|---:|---:|"
    ]

    for group, values in metrics["by_assigner_group"].items():
        model_group = values["model"]
        v0_group = values["v0"]

        lines.append(
            f"| {group} | "
            f"{model_group['critical_capture_at_k']} | "
            f"{v0_group['critical_capture_at_k']} | "
            f"{model_group['hc_capture_at_k']} | "
            f"{v0_group['hc_capture_at_k']} |"
        )

    lines.extend([
        "",
        "## Configured Targets",
        "",
        "| Target | Required | Actual | Met |",
        "|---|---:|---:|---|"
    ])

    for name, target in metrics["target_results"].items():
        lines.append(
            f"| {name} | {target['required']} | "
            f"{target['actual']} | {target['met']} |"
        )

    lines.extend([
        "",
        "## Daily Results",
        "",
        "| Day | CVEs | Critical Captured | Critical Total | HC Captured | HC Total |",
        "|---|---:|---:|---:|---:|---:|"
    ])

    for day in metrics["daily_results"]:
        lines.append(
            f"| {day['day']} | {day['cve_count']} | "
            f"{day['critical_captured']} | {day['critical_total']} | "
            f"{day['hc_captured']} | {day['hc_total']} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Historical training data contains only CVEs that received severity scores.",
        "- Model predictions are prioritization estimates, not CVSS scores.",
        "- Performance depends on the historical data and selected features.",
        "- Capture metrics depend on the configured daily review capacity.",
        ""
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
