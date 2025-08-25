import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/transformer_attack_chain_model.keras")
AUTO_LABELED_FILE = os.path.join(os.path.dirname(__file__), "../dataset/auto_labeled_logs.json")
ORIGINAL_DATASET_FILE = os.path.join(os.path.dirname(__file__), "../dataset/original_dataset.json")  # 你的原始 dataset

def retrain():
    """
    將 auto_labeled_logs.json 與原始 dataset 合併 retrain 模型
    """
    print("[INFO] 開始 retrain 模型...")
    # 載入原始 dataset
    with open(ORIGINAL_DATASET_FILE, "r") as f:
        original_data = json.load(f)
    X_orig = np.array([d["features"] for d in original_data])
    y_orig = np.array([d["label"] for d in original_data])

    # 載入自動標記 dataset
    if os.path.exists(AUTO_LABELED_FILE):
        with open(AUTO_LABELED_FILE, "r") as f:
            auto_data = json.load(f)
        X_auto = np.array([d["features"] for d in auto_data])
        y_auto = np.array([d["label"] for d in auto_data])
        X = np.vstack([X_orig, X_auto])
        y = np.hstack([y_orig, y_auto])
    else:
        X, y = X_orig, y_orig

    # 載入現有模型
    model = load_model(MODEL_PATH)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=3, batch_size=32, verbose=1)  # 可調整 epoch
    model.save(MODEL_PATH)
    print("[INFO] 模型 retrain 完成並儲存")
