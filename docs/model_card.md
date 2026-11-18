@'
# CVE Triage Assistant Model Card

## Intended use

This four-class Logistic Regression model uses TF-IDF description features and categorical CVE metadata to help order CVEs for analyst review. The ranking also puts configured never-wait CWE cases first. It supports prioritization; its probability is not a CVSS score and does not replace a severity assessment.

## Data and labels

The training data contains CVE records with severity labels from the available publisher or CISA scoring information. Severity is the target label: LOW, MEDIUM, HIGH, or CRITICAL. CVEs without an available severity label are not represented in this training history.

The data profile reports 77,431 CVEs published in 2024–2025. The model uses the description, assigner, assigner group, vendor, and product fields. Score-related fields and other excluded columns are not model features.

## Evaluation design

Data was split by publication time without shuffling:

- Training: before September 2025
- Validation: September–October 2025
- Test: November–December 2025

Model selection used the validation period. The final figures below are from the test period. The test evaluation removed 49 records whose normalized descriptions also appeared in training. Daily review capacity was K = 40.

## Test results

| Metric | Model | V0 baseline |
|---|---:|---:|
| Critical capture@K, overall | 0.7395 | 0.4189 |
| High/Critical capture@K, overall | 0.4103 | 0.3087 |
| Critical capture@K, other assigners | 0.7225 | 0.1329 |
| Critical capture@K, top 10 assigners | 0.7473 | 0.5505 |
| High/Critical capture@K, other assigners | 0.4307 | 0.1203 |
| High/Critical capture@K, top 10 assigners | 0.4016 | 0.3896 |

Critical-class recall was 0.5865. The test period covered 61 days; 16 days had 40 or fewer CVEs.

## Limitations

The training history includes only CVEs that received a severity score, so it may not represent all newly published vulnerabilities. Performance depends on the historical data, selected features, and daily review capacity. Ranking scores are estimates for triage; they are not CVSS scores and must not be used to hide or close a CVE.
'@ | Set-Content -LiteralPath .\docs\model_card.md -Encoding utf8