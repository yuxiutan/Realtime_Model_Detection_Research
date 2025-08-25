from fastapi import FastAPI
from .inference import predict
from .scheduler import start_scheduler_thread

app = FastAPI(title="Fully Automatic Transformer Detection API")

@app.post("/predict")
def run_inference(data: dict):
    return {"result": predict([data["features"]])}

# 啟動背景自動 retrain
start_scheduler_thread()
