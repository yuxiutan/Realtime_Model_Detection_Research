import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/transformer_attack_chain_model.keras")
model = load_model(MODEL_PATH)

def predict(features_batch):
    """
    features_batch: list of list
    回傳每個輸入的 softmax 預測結果
    """
    x = np.array(features_batch)
    preds = model.predict(x)
    # 可加入 confidence 判斷
    results = []
    for p in preds:
        pred_class = int(np.argmax(p))
        confidence = float(np.max(p))
        results.append({"predicted_class": pred_class, "confidence": confidence})
    return results

def reload_model():
    global model
    model = load_model(MODEL_PATH)
    print("[INFO] 模型已重新載入")
