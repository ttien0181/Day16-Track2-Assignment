import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_benchmark():
    # 1. Load data
    t0 = time.perf_counter()
    df = pd.read_csv("creditcard.csv")
    data_load_seconds = time.perf_counter() - t0

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2. Split data (Train 60%, Val 20%, Test 20%)
    # Tách lần 1: 80% Train+Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Tách lần 2: 25% của 80% (tương đương 20% tổng) làm Validation
    # Giải thích: 0.25 * 0.80 = 0.20 -> Giữ nguyên tỷ lệ Train 60%, Val 20%, Test 20%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # 3. Training với Early Stopping trên tập Validation
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

    t_train_start = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=callbacks
    )
    training_seconds = time.perf_counter() - t_train_start

    best_iteration = model.best_iteration_ if model.best_iteration_ else model.n_estimators

    # 4. Đánh giá trên tập Test
    # Lưu ý: AUC-ROC nhận xác suất (y_prob), các metrics khác dùng nhãn dự đoán (y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    # 5. Đo Latency (1 dòng)
    sample_row = X_test.iloc[[0]]
    # Warm-up
    for _ in range(10):
        _ = model.predict_proba(sample_row)

    n_single = 200
    t_single_start = time.perf_counter()
    for _ in range(n_single):
        _ = model.predict_proba(sample_row)
    latency_1_row_ms = ((time.perf_counter() - t_single_start) / n_single) * 1000.0

    # 6. Đo Throughput (Batch 1.000 dòng)
    batch_1000 = X_test.iloc[:1000]
    n_batch = 50
    t_batch_start = time.perf_counter()
    for _ in range(n_batch):
        _ = model.predict_proba(batch_1000)
    total_batch_time = time.perf_counter() - t_batch_start
    throughput_1000_rows_per_second = (1000 * n_batch) / total_batch_time

    # 7. Tổng hợp JSON kết quả
    result = {
        "data_load_seconds": round(data_load_seconds, 4),
        "training_seconds": round(training_seconds, 4),
        "best_iteration": int(best_iteration),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "latency_1_row_ms": round(latency_1_row_ms, 4),
        "throughput_1000_rows_per_second": round(throughput_1000_rows_per_second, 2)
    }

    with open("benchmark_result.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\n--- BENCHMARK COMPLETED ---")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    run_benchmark()
