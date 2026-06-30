# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install -U pip && pip install -e ".[openai,anthropic,redis,vector]"

# Copy the rest (evals, scripts, docs).
COPY . .

EXPOSE 8000
CMD ["ai-harness", "serve", "--host", "0.0.0.0", "--port", "8000"]
