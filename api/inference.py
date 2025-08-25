# api/inference.py
import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from Model.inference import (
    preprocess_new_data,
    predict_new_data,
    focal_loss,
    send_discord_alert,
    send_wazuh_alert
)

app = FastAPI(title="Transformer Attack Chain Detection API")

# --- Load model and preprocessors on startup ---
MODEL_PATH = os.path.join('Model', 'improved_transformer_model.keras')
PREPROCESSOR_PATH = os.path.join('Model', 'improved_preprocessors.pkl')

try:
    transformer_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
    )
    print("Transformer model loaded successfully")
except Exception as e:
    print(f"Failed to load Transformer model: {e}")
    raise

try:
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessors = pickle.load(f)
        tokenizer = preprocessors['tokenizer']
        scaler = preprocessors['scaler']
        vectorizer = preprocessors['vectorizer']
    print("Preprocessors loaded successfully")
except Exception as e:
    print(f"Failed to load preprocessors: {e}")
    raise

# --- Load environment variables ---
WAZUH_API_URL = os.getenv("WAZUH_API_URL")
WAZUH_API_USERNAME = os.getenv("WAZUH_API_USERNAME")
WAZUH_API_PASSWORD = os.getenv("WAZUH_API_PASSWORD")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not all([WAZUH_API_URL, WAZUH_API_USERNAME, WAZUH_API_PASSWORD, DISCORD_WEBHOOK_URL]):
    raise ValueError("Missing one or more required environment variables")

# --- Pydantic model for input logs ---
class LogData(BaseModel):
    logs: list  # list of dicts, same format as JSONL

# --- API endpoint ---
@app.post("/predict")
def predict(log_data: LogData):
    df = pd.DataFrame(log_data.logs)
    try:
        results, processed_df = predict_new_data(df, transformer_model, tokenizer, scaler, vectorizer)

        # --- Alerting ---
        if not results.empty:
            most_frequent_chain = results['predicted_chain'].value_counts().idxmax()
            avg_confidence = results['confidence'].mean()
            if avg_confidence > 0.7:  # threshold for high-confidence alert
                unique_agent_names = df['agent.name'].dropna().unique().tolist()
                unique_agent_ips = df['agent.ip'].dropna().unique().tolist()
                alert_message = (
                    f"⚠️ High Confidence Attack Chain Detected ⚠️\n"
                    f"Most frequent attack chain category: {most_frequent_chain}\n"
                    f"Average confidence: {avg_confidence:.4f}\n"
                    f"Number of detected logs: {len(results)}\n"
                    f"Detected unique Agent names: {', '.join(str(n) for n in unique_agent_names) if unique_agent_names else 'None'}\n"
                    f"Detected unique Agent IP addresses: {', '.join(str(ip) for ip in unique_agent_ips) if unique_agent_ips else 'None'}\n"
                    f"Time: {pd.Timestamp.now()}"
                )
                # Send alerts
                send_discord_alert(DISCORD_WEBHOOK_URL, alert_message)
                agent_id = df['agent.id'].dropna().iloc[0] if 'agent.id' in df and not df['agent.id'].dropna().empty else "000"
                send_wazuh_alert(
                    wazuh_api_url=WAZUH_API_URL,
                    username=WAZUH_API_USERNAME,
                    password=WAZUH_API_PASSWORD,
                    alert_message=alert_message,
                    agent_id=agent_id,
                    rule_id="100005",
                    level=12
                )

        # --- Save CSV report ---
        output_report_dir = os.path.join('Model', 'Report')
        os.makedirs(output_report_dir, exist_ok=True)
        csv_output_path = os.path.join(output_report_dir, 'prediction_results.csv')
        results.to_csv(csv_output_path, index=False)

        # --- Save confidence distribution plot ---
        plt.figure(figsize=(10, 6))
        confidence_values = results['confidence']
        plt.hist(confidence_values, bins=20, color='skyblue', alpha=0.7, label='Confidence Distribution')
        kde = gaussian_kde(confidence_values)
        x_range = np.linspace(min(confidence_values), max(confidence_values), 100)
        kde_values = kde(x_range) * len(confidence_values) * (max(confidence_values) - min(confidence_values)) / 20
        plt.plot(x_range, kde_values, 'b-', lw=2, label='KDE')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_output_path = os.path.join(output_report_dir, 'prediction_confidence_distribution.png')
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return results.to_dict(orient='records')

    except Exception as e:
        return {"error": str(e)}
