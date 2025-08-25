import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import shutil
import glob

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../Model/improved_transformer_model.keras")
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "../Model/backup")
AUTO_LABELED_FILE = os.path.join(os.path.dirname(__file__), "../data/auto_labeled_logs.json")
ORIGINAL_DATASET_FILES = [
    os.path.join(os.path.dirname(__file__), "../data/attack_chain_0.jsonl"),
    os.path.join(os.path.dirname(__file__), "../data/attack_chain_1.jsonl"),
    os.path.join(os.path.dirname(__file__), "../data/attack_chain_2.jsonl")
]

def backup_model():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"improved_transformer_model_{timestamp}.keras")
    shutil.copy2(MODEL_PATH, backup_path)
    # 只保留最新5個
    backups = sorted(glob.glob(os.path.join(BACKUP_DIR, "*.keras")), reverse=True)
    for old_backup in backups[5:]:
        os.remove(old_backup)
    print(f"[INFO] 模型已備份: {backup_path}")

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
    backup_model()
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
