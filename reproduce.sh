#!/usr/bin/env bash
# Відтворення експерименту однією командою: ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

python3 scripts/get_data.py                      # дані + перевірка MD5 з DVC-файлів
export GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

docker compose build
docker compose up -d mlflow                      # сервер трекінгу
docker compose run --rm experiment               # обчислювальний експеримент

echo "Готово. Результати: http://localhost:5000 (експеримент reproducibility-benchmark)"
echo "Зупинити сервер: docker compose down"
