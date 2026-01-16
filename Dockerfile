# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (Optional, aber oft hilfreich) Basis-Build-Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies zuerst (Cache-freundlich)
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir .

# Projektcode + Daten
COPY src ./src
COPY data ./data

EXPOSE 8501

# Streamlit starten
CMD ["streamlit", "run", "src/rotor_owl/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
