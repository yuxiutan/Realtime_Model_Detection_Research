from fastapi import FastAPI
from .inference import predict
from .scheduler import start_scheduler_thread

app = FastAPI(title="Realtime Transformer Attack Chain API")

@app.post("/predict")
def run_inference(data: dict):
    """
    data = {"features": [f1, f2, f3, f4, f5, f6]}
    """
    return {"result": predict([data["features"]])}

# 啟動背景自動 retrain
start_scheduler_thread()
