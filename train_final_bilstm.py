import numpy as np
import pandas as pd
import random
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================
# CẤU HÌNH CHUNG
# ==========================
BASE_PREFIX = "InFlam_full"
FP_LIST = ["ecfp", "maccs", "rdkit"]   # chỉ train 3 fingerprint dùng cho web

EPOCHS = 30
BATCH_SIZE = 32
SEED = 42


# ==========================
# 1. Định nghĩa BiLSTM model
# ==========================
def build_model(input_dim: int):
    """
    Xây dựng BiLSTM model đúng như code bạn đang dùng:
    input_shape = (1, input_dim)
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(1, input_dim))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def set_global_seed(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ==========================
# 2. Train + save BiLSTM cho 1 fingerprint
# ==========================
def train_and_save_bilstm_for_fp(fp_name: str):
    """
    Train BiLSTM cho fingerprint fp_name (ecfp/maccs/rdkit)
    trên toàn bộ dữ liệu (train + test) và lưu ra file .keras
    """
    print("\n======================================")
    print(f"Training BiLSTM for fingerprint: {fp_name.upper()}")
    print("======================================")

    # 2.1. Load dữ liệu
    x_train = pd.read_csv(f"{BASE_PREFIX}_x_train_{fp_name}.csv", index_col=0).values
    x_test  = pd.read_csv(f"{BASE_PREFIX}_x_test_{fp_name}.csv", index_col=0).values
    y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
    y_test  = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values.ravel()

    # Gộp và ÉP KIỂU sang float32 cho LSTM
    X_full = np.vstack([x_train, x_test]).astype(np.float32)
    y_full = np.concatenate([y_train, y_test]).astype(np.float32)

    n_samples, n_features = X_full.shape
    print(f"Total samples: {n_samples}")
    print(f"Num features for {fp_name}: {n_features}")
    print(f"dtype X_full: {X_full.dtype}, dtype y_full: {y_full.dtype}")

    # 2.2. Reshape theo đúng format: (batch, timesteps, features)
    # Ở đây timesteps = 1, features = n_features
    X_full_reshaped = X_full.reshape((n_samples, 1, n_features))

    # 2.3. Set seed cho reproducibility
    set_global_seed(SEED)

    # 2.4. Build + train model
    model = build_model(input_dim=n_features)
    history = model.fit(
        X_full_reshaped,
        y_full,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,  # giữ lại 20% từ full data làm validation
        verbose=1,
    )

    print("\nTraining loss/val_loss (last 5 epochs):")
    num_epochs = len(history.history["loss"])
    for epoch in range(max(0, num_epochs - 5), num_epochs):
        train_loss = history.history["loss"][epoch]
        val_loss = history.history["val_loss"][epoch]
        print(
            f"  Epoch {epoch + 1:02d}: "
            f"loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
        )

    # 2.5. Dự đoán sơ bộ trên full data để kiểm tra
    y_prob_full = model.predict(X_full_reshaped, verbose=0).ravel()
    print("Example probabilities (first 5):", y_prob_full[:5])

    # 2.6. Lưu model .keras
    model_filename = f"bilstm_{fp_name}.keras"
    model.save(model_filename)
    print(f"✅ Đã lưu BiLSTM model ({fp_name.upper()}) vào file: {model_filename}")


# ==========================
# 3. MAIN
# ==========================
def main():
    for fp in FP_LIST:
        train_and_save_bilstm_for_fp(fp)


if __name__ == "__main__":
    main()
