import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../Model/improved_transformer_model.keras")
model = load_model(MODEL_PATH)

def predict(features_batch):
    """
    features_batch: list of list
    回傳每個輸入的 softmax 預測結果
    """
    x = np.array(features_batch)
    preds = model.predict(x)
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
