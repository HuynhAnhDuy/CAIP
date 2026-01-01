import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier

BASE_PREFIX = "InFlam_full"
FP_LIST = ["ecfp", "maccs", "rdkit"]   # dùng cả 3 fingerprint


def train_and_save_xgb_for_fp(fp_name: str):
    """
    Train XGBClassifier cho một fingerprint (ecfp/maccs/rdkit)
    và lưu ra file xgb_model_<fp_name>.pkl
    """
    print(f"\n==============================")
    print(f"Training XGBoost for fingerprint: {fp_name.upper()}")
    print(f"==============================")

    # ===============================
    # 1. Load dữ liệu
    # ===============================
    x_train = pd.read_csv(f"{BASE_PREFIX}_x_train_{fp_name}.csv", index_col=0).values
    x_test  = pd.read_csv(f"{BASE_PREFIX}_x_test_{fp_name}.csv", index_col=0).values
    y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
    y_test  = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values.ravel()

    # Gộp train + test để train final (nếu bạn muốn dùng toàn bộ dữ liệu)
    X_full = np.vstack([x_train, x_test])
    y_full = np.concatenate([y_train, y_test])

    # Tính scale_pos_weight cho toàn bộ dữ liệu
    n_pos = np.sum(y_full == 1)
    n_neg = np.sum(y_full == 0)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    print(f"Total samples: {X_full.shape[0]}")
    print(f"Features (fingerprint={fp_name}): {X_full.shape[1]}")
    print(f"Positives: {n_pos}, Negatives: {n_neg}, scale_pos_weight={scale_pos_weight:.3f}")

    # ===============================
    # 2. Khởi tạo model với cùng hyperparameters
    # ===============================
    xgb_final = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=1,
        random_state=42,       # chọn một seed cố định
        n_jobs=-1,
        tree_method="hist",    # hoặc "gpu_hist" nếu dùng GPU
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    )

    # ===============================
    # 3. Train final model
    # ===============================
    print("Training final XGBoost model on full data...")
    xgb_final.fit(X_full, y_full, verbose=True)

    # (Tuỳ chọn) In performance sơ bộ trên full data
    y_pred = xgb_final.predict(X_full)
    y_prob = xgb_final.predict_proba(X_full)[:, 1]
    print(f"Train size: {X_full.shape[0]}, num features: {X_full.shape[1]}")
    print(f"Example predictions (first 5 probabilities): {y_prob[:5]}")

    # ===============================
    # 4. Lưu model -> xgb_model_<fp>.pkl
    # ===============================
    model_filename = f"xgb_model_{fp_name}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(xgb_final, f)

    print(f"✅ Đã lưu mô hình XGB ({fp_name.upper()}) vào file: {model_filename}")


def main():
    for fp in FP_LIST:
        train_and_save_xgb_for_fp(fp)


if __name__ == "__main__":
    main()
