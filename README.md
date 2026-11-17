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

Create and activate a virtual environment.

### Linux or macOS

```bash
python3 -m venv virenv
source virenv/bin/activate
```

### Windows PowerShell

```powershell
py -3.13 -m venv virenv
.\virenv\Scripts\Activate.ps1
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

### Linux or macOS

```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  cve-triage:0.1.0 \
  --input /data/day_sample.csv \
  --output /data/ranked.csv
```

### Windows PowerShell

```powershell
docker run --rm `
  -v "${PWD}/data:/data" `
  cve-triage:0.1.0 `
  --input /data/day_sample.csv `
  --output /data/ranked.csv
```

## Publish the Docker Image

Create a Docker Hub repository named `cve-triage` under the `salaimeha` account and make it public. The published image is `salaimeha/cve-triage:0.1.0`. Publish the versioned tag; do not use `latest`.

### Windows PowerShell

```powershell
$DockerHubUser = "salaimeha"
$Image = "$DockerHubUser/cve-triage:0.1.0"

docker login --username $DockerHubUser
docker build -t $Image .
docker push $Image
```

### Linux or macOS

```bash
DOCKERHUB_USER="salaimeha"
IMAGE="$DOCKERHUB_USER/cve-triage:0.1.0"

docker login --username "$DOCKERHUB_USER"
docker build -t "$IMAGE" .
docker push "$IMAGE"
```

## Run the Published Docker Image

Place the course-provided `day_sample.csv` in the local `data/` directory.

### Windows PowerShell

```powershell
$DockerHubUser = "salaimeha"
$Image = "$DockerHubUser/cve-triage:0.1.0"

docker run --rm `
  -v "${PWD}/data:/data" `
  $Image `
  --input /data/day_sample.csv `
  --output /data/ranked.csv
```

### Linux or macOS

```bash
DOCKERHUB_USER="salaimeha"
IMAGE="$DOCKERHUB_USER/cve-triage:0.1.0"

docker run --rm \
  -v "$(pwd)/data:/data" \
  "$IMAGE" \
  --input /data/day_sample.csv \
  --output /data/ranked.csv
```

Check the output with the course-provided `check_submission.py`:

```bash
python check_submission.py data/ranked.csv
```

## Output Format

The ranked CSV contains:

- `cve_id`
- `rank`
- `score`
- `model_version`

Rank 1 is the highest-priority CVE for review.

## Model

The system uses a four-class Logistic Regression classifier with TF-IDF text features and categorical features.

The final model is registered in MLflow as `cve-triage`.

The batch-ranking CLI uses the exported model artifact packaged inside the Docker image.