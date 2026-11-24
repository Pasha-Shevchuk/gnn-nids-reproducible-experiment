# Ізольоване середовище відтворюваного експерименту (Linux + Python 3.10)
FROM python:3.10-slim

# Хеш Git-коміту передається під час збірки і потрапляє в теги MLflow
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

WORKDIR /app

# Встановлення системних утиліт
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Інсталяція Python-залежностей із закріпленими версіями
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання коду експерименту (дані монтуються як том: -v ./data:/app/data:ro)
COPY train.py .

CMD ["python", "train.py"]
