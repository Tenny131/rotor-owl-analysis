# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Zuerst Metadaten + Source kopieren, damit "pip install ." das src/ sieht
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir .

# Daten (Ontologien) separat
COPY data ./data

EXPOSE 8501

CMD ["streamlit", "run", "src/rotor_owl/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
