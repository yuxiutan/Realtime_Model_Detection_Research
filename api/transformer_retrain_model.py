import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../Model/improved_transformer_model.keras")
AUTO_LABELED_FILE = os.path.join(os.path.dirname(__file__), "../data/auto_labeled_logs.json")
ORIGINAL_DATASET_FILES = [
    os.path.join(os.path.dirname(__file__), "../data/attack_chain_0.jsonl"),
    os.path.join(os.path.dirname(__file__), "../data/attack_chain_1.jsonl"),
    os.path.join(os.path.dirname(__file__), "../data/attack_chain_2.jsonl")
]

def load_original_dataset():
    X_list, y_list = [], []
    for fpath in ORIGINAL_DATASET_FILES:
        with open(fpath, "r") as f:
            for line in f:
                log = json.loads(line)
                X_list.append(log["features"])
                y_list.append(log["label"])
    return np.array(X_list), np.array(y_list)

def retrain():
    print("[INFO] 開始 retrain 模型...")
    X_orig, y_orig = load_original_dataset()
    if os.path.exists(AUTO_LABELED_FILE):
        with open(AUTO_LABELED_FILE, "r") as f:
            auto_data = json.load(f)
        if auto_data:
            X_auto = np.array([d["features"] for d in auto_data])
            y_auto = np.array([d["label"] for d in auto_data])
            X = np.vstack([X_orig, X_auto])
            y = np.hstack([y_orig, y_auto])
        else:
            X, y = X_orig, y_orig
    else:
        X, y = X_orig, y_orig

    model = load_model(MODEL_PATH)
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=3, batch_size=32, verbose=1)
    model.save(MODEL_PATH)
    print("[INFO] retrain 完成")
