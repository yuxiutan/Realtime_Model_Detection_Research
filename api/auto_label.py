import os
import json
import numpy as np
from sklearn.cluster import DBSCAN
from threading import Lock

LOCK = Lock()
UNCERTAIN_LOG_FILE = os.path.join(os.path.dirname(__file__), "../data/uncertain_logs.json")
AUTO_LABELED_FILE = os.path.join(os.path.dirname(__file__), "../data/auto_labeled_logs.json")

if not os.path.exists(AUTO_LABELED_FILE):
    with open(AUTO_LABELED_FILE, "w") as f:
        json.dump([], f)

def pseudo_labeling():
    if not os.path.exists(UNCERTAIN_LOG_FILE):
        return
    with LOCK:
        with open(UNCERTAIN_LOG_FILE, "r") as f:
            data = json.load(f)
        high_confidence_logs = [log for log in data if log["confidence"] >= 0.6]
        uncertain_logs = [log for log in data if log["confidence"] < 0.6]

        if high_confidence_logs:
            with open(AUTO_LABELED_FILE, "r") as f:
                labeled_data = json.load(f)
            labeled_data.extend([{"features": log["features"], "label": log["predicted_class"]} for log in high_confidence_logs])
            with open(AUTO_LABELED_FILE, "w") as f:
                json.dump(labeled_data, f, indent=2)

        with open(UNCERTAIN_LOG_FILE, "w") as f:
            json.dump(uncertain_logs, f, indent=2)

def cluster_and_label():
    if not os.path.exists(UNCERTAIN_LOG_FILE):
        return
    with LOCK:
        with open(UNCERTAIN_LOG_FILE, "r") as f:
            data = json.load(f)
        if not data:
            return

        X = np.array([log["features"] for log in data])
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
        labels = clustering.labels_

        with open(AUTO_LABELED_FILE, "r") as f:
            auto_data = json.load(f)
        for log, cluster_label in zip(data, labels):
            if cluster_label != -1:
                auto_data.append({"features": log["features"], "label": int(cluster_label)})
        with open(AUTO_LABELED_FILE, "w") as f:
            json.dump(auto_data, f, indent=2)

        remaining = [log for log, lbl in zip(data, labels) if lbl == -1]
        with open(UNCERTAIN_LOG_FILE, "w") as f:
            json.dump(remaining, f, indent=2)
