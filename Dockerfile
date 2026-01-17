# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG REPO_URL="https://github.com/Tenny131/rotor-owl-analysis.git"
ARG REPO_REF="main"

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir streamlit \
 && pip install --no-cache-dir "rotor-owl-analysis @ git+${REPO_URL}@${REPO_REF}"

EXPOSE 8501

CMD ["sh", "-c", "streamlit run $(python -c \"import rotor_owl.streamlit_app as m; print(m.__file__)\" ) --server.address=0.0.0.0 --server.port=8501"]
