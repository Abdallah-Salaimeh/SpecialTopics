
# SpecialTopics — CVE Triage Assistant

A machine learning system that prioritizes Common Vulnerabilities and Exposures (CVEs) for security analyst review.

## Team

- Abdallah Salaimeh
- Dana Kharouf

## Project Scope

The system estimates the probability that a CVE belongs to the CRITICAL severity class and ranks CVEs to support security triage.

The prediction is a prioritization estimate, not a replacement for a CVSS score.

## Installation

Python 3.13 is required.

Create and activate a virtual environment:

```bash
python3 -m venv virenv
source virenv/bin/activate
```

Install the project:

```bash
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
pytest -v
```

## Run the Training Pipeline

```bash
python -m specialtopics.pipeline
```

## Run Batch Ranking

The trained model must be available in the local `models/` directory.

```bash
python -m specialtopics.cli \
  --input data/day_sample.csv \
  --output data/ranked.csv
```

## Build the Docker Image

```bash
docker build -t cve-triage:0.1.0 .
```

## Run the Docker Image

```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  cve-triage:0.1.0 \
  --input /data/day_sample.csv \
  --output /data/ranked.csv
```

## Output Format

The ranked CSV contains:

- cve_id
- rank
- score
- model_version

Rank 1 is the highest-priority CVE for review.

## Model

The system uses a four-class Logistic Regression classifier with TF-IDF text features and categorical features.

The final model is registered in MLflow as `cve-triage`.

The batch-ranking CLI uses the exported model artifact packaged inside the Docker image.
