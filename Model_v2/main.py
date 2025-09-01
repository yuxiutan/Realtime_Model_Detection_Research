import os
import json
import datetime
import requests
import urllib3
from fastapi import FastAPI, Body
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import shutil
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI()

# Environment variables or defaults
WAZUH_USER = os.getenv('WAZUH_USER', 'admin')
WAZUH_PASSWORD = os.getenv('WAZUH_PASSWORD', 'SecretPassword')
WAZUH_API_URL = os.getenv('WAZUH_API_URL', 'https://100.79.144.59:9200/wazuh-alerts-*/_search')
MODEL_DIR = os.getenv('MODEL_DIR', '/app/model')
DATA_DIR = os.getenv('DATA_DIR', '/app/data')
LOW_CONF_FILE = os.path.join(DATA_DIR, 'low_confidence.json')
ORIGINAL_WEIGHTS = os.path.join(MODEL_DIR, 'original_weights')
LATEST_WEIGHTS_QUEUE = deque(maxlen=3)  # Keep latest 3 versions
REF_EMBEDDINGS = os.path.join(MODEL_DIR, 'reference_embeddings.pkl')
ACC_FILE = os.path.join(MODEL_DIR, 'latest_acc.txt')

# Ensure directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load latest weights queue from disk if exists
QUEUE_FILE = os.path.join(MODEL_DIR, 'weights_queue.pkl')
if os.path.exists(QUEUE_FILE):
    with open(QUEUE_FILE, 'rb') as f:
        LATEST_WEIGHTS_QUEUE = pickle.load(f)
else:
    LATEST_WEIGHTS_QUEUE = deque(maxlen=3)

class LogData(BaseModel):
    logs: list

# Fetch Wazuh data
def fetch_wazuh_data(start_time, end_time):
    data_list = []
    search_after_value = None
    while True:
        query_data = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"timestamp": {"gte": start_time, "lte": end_time}}},
                        {"terms": {"agent.name": ["DESKTOP-66TG6SE", "DESKTOP-66G2GGG", "connector-node"]}}
                    ]
                }
            },
            "_source": ["timestamp", "agent.ip", "agent.name", "agent.id", "rule.id", "rule.mitre.id", "rule.level", "rule.description", "data.srcip", "data.dstip", "full_log"],
            "sort": [{"timestamp": {"order": "asc"}}],
            "size": 10000
        }
        if search_after_value:
            query_data["search_after"] = search_after_value
        try:
            response = requests.post(WAZUH_API_URL, auth=(WAZUH_USER, WAZUH_PASSWORD), json=query_data, verify=False)
            response.raise_for_status()
            hits = response.json().get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                source = hit.get("_source", {})
                processed_log = {
                    "timestamp": source.get("timestamp"),
                    "agent": {"ip": source.get("agent", {}).get("ip"), "name": source.get("agent", {}).get("name"), "id": source.get("agent", {}).get("id")},
                    "rule": {"id": source.get("rule", {}).get("id"), "mitre": {"id": source.get("rule", {}).get("mitre", {}).get("id", "T0000")}, "level": source.get("rule", {}).get("level"), "description": source.get("rule", {}).get("description")},
                    "data": {"srcip": source.get("data", {}).get("srcip"), "dstip": source.get("data", {}).get("dstip")},
                    "full_log": source.get("full_log")
                }
                data_list.append(processed_log)
            search_after_value = hits[-1].get("sort") if hits else None
        except Exception:
            break
    return data_list

# Preprocess
def preprocess_data(logs):
    sequences = []
    for log in logs:
        desc = log.get("rule", {}).get("description", "")
        src_ip = log.get("data", {}).get("srcip", "None")
        dst_ip = log.get("data", {}).get("dstip", "None")
        sequence = f"{desc} from {src_ip} to {dst_ip}"
        sequences.append(sequence)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized = [tokenizer(seq, padding="max_length", truncation=True, max_length=512, return_tensors="pt") for seq in sequences]
    return tokenized, sequences

# Inference
def perform_inference(logs):
    tokenized, sequences = preprocess_data(logs)
    model_path = LATEST_WEIGHTS_QUEUE[-1] if LATEST_WEIGHTS_QUEUE else ORIGINAL_WEIGHTS
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    with open(REF_EMBEDDINGS, "rb") as f:
        ref_data = pickle.load(f)
        four_emb = ref_data["four_embedding"]
        apt_emb = ref_data["apt_embedding"]
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.eval()
    results = []
    low_conf_data = []
    for seq, inputs in zip(sequences, tokenized):
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            pred_label = np.argmax(probs)
            conf = np.max(probs)
            emb = get_embedding(seq, tokenizer, bert_model)
            four_sim = cosine_similarity(emb, four_emb)
            apt_sim = cosine_similarity(emb, apt_emb)
            pred_chain = "FourInOne" if four_sim > apt_sim else "APT29"
        result = {"sequence": seq, "prediction": pred_chain, "confidence": float(conf), "similarities": {"FourInOne": float(four_sim), "APT29": float(apt_sim)}}
        results.append(result)
        if conf < 0.3:
            low_conf_data.append(result)
    if low_conf_data:
        with open(LOW_CONF_FILE, 'a', encoding='utf-8') as f:
            for item in low_conf_data:
                f.write(json.dumps(item) + '\n')
    return results

def get_embedding(sequence, tokenizer, bert_model):
    inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Training
def train_model():
    four_logs = load_logs(os.path.join(DATA_DIR, 'attack_chain_FourInOne.json'))
    apt_logs = load_logs(os.path.join(DATA_DIR, 'attack_chain_APT29.json'))
    four_seq = extract_features(four_logs)
    apt_seq = extract_features(apt_logs)
    synthetic_four = generate_synthetic_sequences(four_seq, 0)
    synthetic_apt = generate_synthetic_sequences(apt_seq, 1)
    all_data = synthetic_four + synthetic_apt
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_data = [tokenizer(item["sequence"], padding="max_length", truncation=True, max_length=512, return_tensors="pt") for item in all_data]
    labels = np.array([item["label"] for item in all_data])
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(tokenized_data, labels, test_size=0.2, random_state=42)
    input_ids = torch.cat([item['input_ids'] for item in train_inputs], dim=0)
    attention_masks = torch.cat([item['attention_mask'] for item in train_inputs], dim=0)
    train_labels_tensor = torch.tensor(train_labels)
    dataset = TensorDataset(input_ids, attention_masks, train_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, hidden_dropout_prob=0.3)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    for epoch in range(3):
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    test_input_ids = torch.cat([item['input_ids'] for item in test_inputs], dim=0)
    test_attention_masks = torch.cat([item['attention_mask'] for item in test_inputs], dim=0)
    test_labels_tensor = torch.tensor(test_labels)
    with torch.no_grad():
        outputs = model(input_ids=test_input_ids, attention_mask=test_attention_masks)
        preds = torch.argmax(outputs.logits, dim=1).numpy()
    acc = accuracy_score(test_labels, preds)
    prev_acc = 0.0
    if os.path.exists(ACC_FILE):
        with open(ACC_FILE, 'r') as f:
            prev_acc = float(f.read())
    if acc > prev_acc:
        new_version = len(LATEST_WEIGHTS_QUEUE) + 1
        new_path = os.path.join(MODEL_DIR, f'version_{new_version}')
        model.save_pretrained(new_path)
        LATEST_WEIGHTS_QUEUE.append(new_path)
        with open(ACC_FILE, 'w') as f:
            f.write(str(acc))
        with open(QUEUE_FILE, 'wb') as f:
            pickle.dump(LATEST_WEIGHTS_QUEUE, f)
        if len(LATEST_WEIGHTS_QUEUE) > 3:
            old_path = LATEST_WEIGHTS_QUEUE.popleft()
            shutil.rmtree(old_path)
    # Update ref embeddings
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    four_embedding = get_embedding(four_seq, tokenizer, bert_model)
    apt_embedding = get_embedding(apt_seq, tokenizer, bert_model)
    with open(REF_EMBEDDINGS, "wb") as f:
        pickle.dump({"four_embedding": four_embedding, "apt_embedding": apt_embedding}, f)

# Helpers
def load_logs(file_path):
    logs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except:
                pass
    return logs

def extract_features(logs):
    sequences = []
    for log in logs:
        desc = log.get("rule", {}).get("description", "")
        src_ip = log.get("data", {}).get("srcip", "None")
        dst_ip = log.get("data", {}).get("dstip", "None")
        sequences.append(f"{desc} from {src_ip} to {dst_ip}")
    return " ".join(sequences)

def generate_synthetic_sequences(base_seq, label, num_samples=120):
    data = []
    for _ in range(num_samples):
        words = base_seq.split()
        np.random.shuffle(words)
        perturbed = " ".join(words[:int(len(words)*0.9)])
        data.append({"sequence": perturbed, "label": label})
    return data

# Periodic fetch task
def periodic_fetch():
    now = datetime.datetime.utcnow()
    start_time = (now - datetime.timedelta(minutes=3)).isoformat() + 'Z'
    end_time = now.isoformat() + 'Z'
    logs = fetch_wazuh_data(start_time, end_time)
    if logs:
        perform_inference(logs)

# Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(periodic_fetch, 'interval', minutes=3)
scheduler.add_job(train_model, CronTrigger(hour=21, minute=0))  # Daily at 21:00 UTC
scheduler.start()

# Initial setup: If no original weights, train once
if not os.path.exists(ORIGINAL_WEIGHTS):
    train_model()
    first_version = os.path.join(MODEL_DIR, 'version_1')
    if os.path.exists(first_version):
        shutil.copytree(first_version, ORIGINAL_WEIGHTS)

# API Endpoint
@app.post("/predict")
def predict(data: LogData = Body(...)):
    results = perform_inference(data.logs)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
