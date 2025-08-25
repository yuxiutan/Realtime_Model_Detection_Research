import time
from threading import Thread
from datetime import datetime
from .auto_label import pseudo_labeling, cluster_and_label
from .train import retrain
from .inference import reload_model
import os
import json

AUTO_LABELED_FILE = os.path.join(os.path.dirname(__file__), "../data/auto_labeled_logs.json")
RETRAIN_INTERVAL_HOURS = 168  # 每週 retrain
RETRAIN_MIN_LOGS = 50

def retrain_if_needed():
    pseudo_labeling()
    cluster_and_label()

    if not os.path.exists(AUTO_LABELED_FILE):
        return
    with open(AUTO_LABELED_FILE, "r") as f:
        data = json.load(f)
    if len(data) >= RETRAIN_MIN_LOGS:
        print(f"[{datetime.now()}] 達到 {len(data)} 筆自動標記 log，開始 retrain...")
        retrain()
        reload_model()
        with open(AUTO_LABELED_FILE, "w") as f:
            json.dump([], f, indent=2)
        print(f"[{datetime.now()}] retrain 完成，已清空 auto_labeled_logs.json")
    else:
        print(f"[{datetime.now()}] 自動標記 log 數 {len(data)} 未達 retrain 門檻 ({RETRAIN_MIN_LOGS})")

def schedule_retrain():
    while True:
        print(f"[{datetime.now()}] 自動 retrain檢查中...")
        retrain_if_needed()
        time.sleep(RETRAIN_INTERVAL_HOURS * 3600)

def start_scheduler_thread():
    thread = Thread(target=schedule_retrain, daemon=True)
    thread.start()
