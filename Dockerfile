
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./

COPY src ./src

RUN python -m pip install --no-cache-dir .

COPY features.yaml ./features.yaml

COPY models ./models

ENTRYPOINT ["python", "-m", "specialtopics.cli"]
