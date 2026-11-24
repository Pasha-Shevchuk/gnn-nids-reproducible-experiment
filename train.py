"""Відтворюваний експеримент: потокова базова модель NIDS на NSL-KDD з трекінгом MLflow.

Модель Random Forest навчається на KDDTrain+ (стратифіковане розбиття 80/20 на Train/Validation)
і додатково перевіряється на KDDTest+ – вибірці поза навчальним розподілом (17 нових типів атак).
Усі гіперпараметри, метрики, хеші даних, версії бібліотек і артефакти фіксуються в MLflow.

Запуск:  python train.py [--n-estimators 100] [--max-depth 20] [--seed 42]
Сервер трекінгу задається змінною MLFLOW_TRACKING_URI (за замовчуванням – локальна тека ./mlruns).
"""
import argparse
import hashlib
import os
import platform
import random
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = os.environ.get("DATA_DIR", "data")
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]
CAT = ["protocol_type", "service", "flag"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Хеш коміту: зі змінної середовища (у Docker) або з локального Git."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load(path: str):
    df = pd.read_csv(path, header=None, names=COLS)
    X = df.drop(columns=["label", "difficulty"])
    y = (df["label"] != "normal").astype(int)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default="rf_baseline")
    args = ap.parse_args()
    set_seed(args.seed)

    mlflow.set_experiment("reproducibility-benchmark")
    with mlflow.start_run(run_name=args.run_name):
        X, y = load(TRAIN_FILE)
        X_ood, y_ood = load(TEST_FILE)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)

        # Гіперпараметри та контекст відтворюваності
        mlflow.log_params({"model": "RandomForestClassifier", "n_estimators": args.n_estimators,
                           "max_depth": args.max_depth, "random_state": args.seed, "test_size": 0.2,
                           "stratify": True, "n_train": len(X_tr), "n_val": len(X_val), "n_ood": len(X_ood)})
        mlflow.set_tags({"dataset": "NSL-KDD", "task": "binary NIDS (normal vs attack)",
                         "git_commit": git_commit(), "md5_train": md5(TRAIN_FILE), "md5_test": md5(TEST_FILE),
                         "python": platform.python_version(), "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__, "mlflow": mlflow.__version__})

        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
        model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed, n_jobs=1))])
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        mlflow.log_metric("train_time_s", time.perf_counter() - t0)

        # Метрики якості: валідація (той самий розподіл) і KDDTest+ (поза розподілом)
        y_val_pred = model.predict(X_val)
        y_ood_pred = model.predict(X_ood)
        p_ood = model.predict_proba(X_ood)[:, 1]
        mlflow.log_metrics({
            "f1_val_macro": f1_score(y_val, y_val_pred, average="macro"),
            "f1_val_weighted": f1_score(y_val, y_val_pred, average="weighted"),
            "f1_ood_macro": f1_score(y_ood, y_ood_pred, average="macro"),
            "precision_ood_macro": precision_score(y_ood, y_ood_pred, average="macro"),
            "recall_ood_macro": recall_score(y_ood, y_ood_pred, average="macro"),
            "roc_auc_ood": roc_auc_score(y_ood, p_ood),
        })

        # Артефакти: матриця помилок, текстовий звіт, модель із сигнатурою
        os.makedirs("outputs", exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(y_ood, y_ood_pred, display_labels=["normal", "attack"],
                                                cmap="Blues", values_format="d", ax=ax, colorbar=False)
        ax.set_title("Confusion matrix, KDDTest+")
        fig.tight_layout()
        fig.savefig("outputs/confusion_matrix_ood.png", dpi=150)
        with open("outputs/classification_report_ood.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_ood, y_ood_pred, target_names=["normal", "attack"], digits=4))
        mlflow.log_artifacts("outputs", artifact_path="reports")
        mlflow.sklearn.log_model(model, "random_forest_model",
                                 signature=infer_signature(X_val.head(100), y_val_pred[:100]),
                                 input_example=X_val.head(3))
        print("Run finished:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
