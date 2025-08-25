import schedule
import time
import os
import json
from threading import Thread
from datetime import datetime
from .auto_label import pseudo_labeling, cluster_and_label
from .transformer_retrain_model import retrain
from .inference import reload_model

AUTO_LABELED_FILE = os.path.join(os.path.dirname(__file__), "../data/auto_labeled_logs.json")
LOCK_FILE = os.path.join(os.path.dirname(__file__), "retrain.lock")
RETRAIN_MIN_LOGS = 500000  # 調整為50萬筆

def retrain_if_needed():
    if os.path.exists(LOCK_FILE):
        print(f"[{datetime.now()}] retrain 正在執行，略過本次")
        return

    pseudo_labeling()
    cluster_and_label()

    if not os.path.exists(AUTO_LABELED_FILE):
        return
    with open(AUTO_LABELED_FILE, "r") as f:
        data = json.load(f)

    if len(data) >= RETRAIN_MIN_LOGS:
        print(f"[{datetime.now()}] 達到 {len(data)} 筆自動標記 log，開始 retrain...")
        try:
            with open(LOCK_FILE, "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            retrain()
            reload_model()

            # 清空 auto_labeled_logs
            with open(AUTO_LABELED_FILE, "w") as f:
                json.dump([], f, indent=2)

        finally:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        print(f"[{datetime.now()}] retrain 完成")
    else:
        print(f"[{datetime.now()}] 自動標記 log 數 {len(data)} 未達 retrain 門檻 ({RETRAIN_MIN_LOGS})")

def schedule_retrain():
    schedule.every().day.at("21:00").do(retrain_if_needed)
    while True:
        schedule.run_pending()
        time.sleep(60)

def start_scheduler_thread():
    thread = Thread(target=schedule_retrain, daemon=True)
    thread.start()
