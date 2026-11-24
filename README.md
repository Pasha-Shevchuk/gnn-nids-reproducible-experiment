# Reproducible Benchmark Environment for ML Experiments
### Відтворюваний експеримент: потокова базова модель NIDS на NSL-KDD

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![MLflow 2.11](https://img.shields.io/badge/MLflow-2.11.0-0194E2.svg)
![DVC](https://img.shields.io/badge/data-DVC-945DD6.svg)

Артефактний пакет до практичної роботи № 6 «Створення середовища відтворюваного експерименту та публікація артефактів».
Автор: **Шевчук Павло Олександрович**, група КНм-26-1, Хмельницький національний університет.

Експеримент є частиною дослідження *«Виявлення мережевих вторгнень на основі графових нейронних мереж з урахуванням
міждоменного узагальнення»*. Тут навчається потокова базова модель (Random Forest), з якою надалі порівнюються графові
моделі. Модель оцінюється двічі: на валідаційній вибірці з того самого розподілу та на KDDTest+ – наборі з 17 типами
атак, яких немає в навчальних даних.

## Що забезпечує відтворюваність

| Рівень | Інструмент | Файл |
|---|---|---|
| Код і його версії | Git, теги релізів | `train.py`, тег `v1.0.0` |
| Залежності | закріплені версії пакетів | `requirements.txt` |
| ОС і середовище виконання | Docker (`python:3.10-slim`) | `Dockerfile`, `.dockerignore` |
| Сервіс трекінгу + експеримент | Docker Compose | `docker-compose.yml` |
| Дані | DVC (MD5-хеші в Git) | `data/*.dvc`, `scripts/get_data.py` |
| Параметри, метрики, моделі | MLflow Tracking | `train.py` |
| Випадковість | `random_state = 42`, `PYTHONHASHSEED = 42` | `train.py`, `Dockerfile` |
| Метадані для DOI (FAIR) | Zenodo, Citation File Format | `zenodo.json`, `.zenodo.json`, `CITATION.cff` |
| Ліцензія | MIT | `LICENSE` |

## Структура репозиторію

```
.
├── Dockerfile               # ізольоване середовище Linux + Python 3.10
├── docker-compose.yml       # сервер MLflow + контейнер експерименту
├── train.py                 # експеримент із логуванням у MLflow
├── requirements.txt         # закріплені залежності (для контейнера)
├── requirements-dev.txt     # DVC для роботи з даними на хості
├── reproduce.sh             # відтворення однією командою
├── scripts/get_data.py      # завантаження NSL-KDD і перевірка MD5 за DVC-файлами
├── data/
│   ├── KDDTrain+.txt.dvc    # вказівник DVC (MD5 + розмір), сам файл – не в Git
│   └── KDDTest+.txt.dvc
├── zenodo.json / .zenodo.json
├── CITATION.cff
└── LICENSE
```

## Швидкий старт (одна команда)

Потрібні лише Git, Python 3 і Docker (Docker Desktop або Docker Engine з плагіном Compose).

```bash
git clone https://github.com/Pasha-Shevchuk/gnn-nids-reproducible-experiment.git
cd gnn-nids-reproducible-experiment
bash reproduce.sh
```

Скрипт завантажить дані й перевірить їхні хеші, збере образ, підніме сервер MLflow і виконає експеримент.
Результати – у вебінтерфейсі **http://localhost:5000**, експеримент `reproducibility-benchmark`.
Зупинити сервер: `docker compose down`.

## Покрокове відтворення

**1. Клонування та дані**

```bash
git clone https://github.com/Pasha-Shevchuk/gnn-nids-reproducible-experiment.git
cd gnn-nids-reproducible-experiment
git checkout v1.0.0                 # точна версія коду з релізу
python3 scripts/get_data.py         # NSL-KDD з публічного дзеркала + перевірка MD5
```

Очікуваний вивід: `md5 ... – OK` для обох файлів. Якщо встановлено DVC (`pip install -r requirements-dev.txt`),
відповідність даних фіксованій версії також перевіряє `dvc status` (`Data and pipelines are up to date`).

**2а. Один контейнер (Етап 4 практичної роботи)**

```bash
docker build -t cs-experiment:v1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
docker run --rm -v "$(pwd)/mlruns:/app/mlruns" -v "$(pwd)/data:/app/data:ro" cs-experiment:v1.0
mlflow ui --port 5000                # або: pip install mlflow==2.11.0
```

**2б. Docker Compose (сервер MLflow + експеримент)**

```bash
export GIT_COMMIT=$(git rev-parse HEAD)
docker compose build
docker compose up -d mlflow
docker compose run --rm experiment
```

Сервер MLflow зберігає метадані в SQLite (`mlflow_server/mlflow.db`), а артефакти – у `mlflow_server/artifacts`;
контейнер експерименту передає їх через HTTP-проксі сервера.

**3. Без Docker (локальне віртуальне середовище)**

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --n-estimators 100 --max-depth 20 --seed 42
mlflow ui --port 5000
```

## Що фіксує MLflow

- **Параметри:** модель, `n_estimators`, `max_depth`, `random_state`, частка валідації, розміри вибірок.
- **Метрики:** `f1_val_macro`, `f1_val_weighted`, `f1_ood_macro`, `precision_ood_macro`, `recall_ood_macro`,
  `roc_auc_ood`, `train_time_s`.
- **Теги:** хеш Git-коміту, MD5 обох файлів даних, версії Python, scikit-learn, pandas, numpy, MLflow.
- **Артефакти:** матриця помилок на KDDTest+, текстовий звіт класифікації, модель (`mlflow.sklearn`) із сигнатурою та
  прикладом входу.

## Еталонні результати (v1.0.0, seed = 42)

| Метрика | Значення |
|---|---|
| Макро-F1, Validation | 0.9989 |
| Макро-F1, KDDTest+ (OOD) | 0.7756 |
| Precision (макро), KDDTest+ | 0.8163 |
| Recall (макро), KDDTest+ | 0.8004 |
| ROC-AUC, KDDTest+ | 0.9678 |
| Час навчання | ≈ 6–9 с на 1 ядрі CPU (залежить від апаратури) |

**Перевірка повторюваності (ACM Repeatability):** два запуски (`rf_baseline` і `rf_repeatability_check`) у тому самому
середовищі дали тотожні до останнього знака метрики якості; відрізняється лише час навчання. Ті самі значення отримано й
у «чистому» віртуальному середовищі, встановленому лише з `requirements.txt`, та в режимі «сервер MLflow + HTTP-клієнт»,
який використовує docker-compose.

Розрив між Validation (0.9989) і KDDTest+ (0.7756) – це проблема узагальнення на нові типи атак, якій присвячене основне
дослідження.

Знімки вебінтерфейсу MLflow з цими запусками (список запусків, параметри й метрики, артефакт матриці помилок)
наведено в PDF-звіті до практичної роботи № 6.

## Відповідність принципам FAIR

- **Findable** – метадані в `zenodo.json`; після архівування релізу в Zenodo пакет отримує DOI.
- **Accessible** – відкритий репозиторій і відкритий доступ (`access_right: open`).
- **Interoperable** – відкриті формати: CSV/TXT для даних, YAML/JSON для метаданих, формат MLflow Model для моделі.
- **Reusable** – ліцензія MIT, закріплені залежності, інструкція відтворення, `CITATION.cff`.

## Публікація в Zenodo з DOI

1. Увійти на zenodo.org через GitHub і в розділі *GitHub* увімкнути цей репозиторій.
2. Створити реліз на GitHub із тегом `v1.0.0`. Zenodo автоматично заархівує його і присвоїть DOI, узявши метадані з
   `.zenodo.json` (копія файлу `zenodo.json`).

## Дані

NSL-KDD: Tavallaee M., Bagheri E., Lu W., Ghorbani A.A. *A detailed analysis of the KDD CUP 99 data set* // IEEE CISDA,
2009. DOI: [10.1109/CISDA.2009.5356528](https://doi.org/10.1109/CISDA.2009.5356528). Датасет не поширюється в цьому
репозиторії: файли завантажуються з публічного дзеркала, а їхня цілісність перевіряється за MD5-хешами, зафіксованими DVC.

## Цитування та ліцензія

Код поширюється за ліцензією [MIT](LICENSE). Як цитувати – див. [`CITATION.cff`](CITATION.cff).
