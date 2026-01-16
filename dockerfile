# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (falls OWL libs / XML etc. gebraucht werden)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies zuerst (besserer Layer-Cache)
COPY pyproject.toml README.md ./
# falls du ein lockfile hast, mitkopieren:
# COPY uv.lock ./
# COPY poetry.lock ./

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir .

# Projektcode
COPY src ./src
COPY data ./data

EXPOSE 8501

# Streamlit starten (Passe den Pfad an, falls deine App woanders liegt)
CMD ["streamlit", "run", "src/rotor_owl/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
