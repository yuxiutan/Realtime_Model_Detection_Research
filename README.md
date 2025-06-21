# Realtime_LSTM_Chain_Detection

## 📌 Project Overview

This project aims to detect abnormal behaviors in industrial OT environments by training an LSTM model with logs collected from **Wazuh** via API. It identifies potential multi-step attack chains and triggers alerts based on similarity scores. A real-time monitoring dashboard is provided using **Dash**.

## 📂 Project Structure

```bash
LSTM-Behavior-Detection/
├── analysis/
│   ├── classification_report.csv
│   └── output_training_metrics.json
├── classes/
│   ├── agent_id_classes.npy
│   ├── agent_ip_classes.npy
│   ├── agent_name_classes.npy
│   ├── attack_chain_classes.npy
│   ├── eventdata_image_classes.npy
│   ├── mitre_id_classes.npy
│   └── rule_id_classes.npy
├── data/
│   └── generated_wazuh_logs.jsonl
├── model/
│   ├── attack_chain_vectors.npy
│   ├── build_model.py         # Train and evaluate the LSTM model
│   └── lstm_attack_chain_model.keras
├── preprocessing/
│   └── prepare_dataset.py     # Label encoding, padding, slicing
├── utils/
│   └── wazuh_api.py
├── Realtime_detection.py      # Run real-time inference and compute similarity
├── app.py                     # Dash-based real-time dashboard
├── encoders.pkl
├── model_info.txt             
├── requirements.txt           # List of dependencies
├── generated_wazuh_logs.jsonl # Raw training dataset from Wazuh API

