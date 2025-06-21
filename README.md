# Realtime_LSTM_Chain_Detection

**License:** MIT  
**Author:** [Yuxiu Tan](https://github.com/yuxiutan)

## 📌 Project Overview

This project aims to detect abnormal behaviors in industrial OT environments by training an LSTM model with logs collected from **Wazuh** via API. It identifies potential multi-step attack chains and triggers alerts based on similarity scores. A real-time monitoring dashboard is provided using **Dash**.

## 📂 Project Structure

```bash
Realtime_LSTM_Chain_Detection/
├── analysis/                        # Model evaluation outputs
│ ├── classification_report.csv      # Precision, Recall, F1-score per class
│ └── output_training_metrics.json   # Accuracy/loss logs per epoch
├── classes/                         # LabelEncoder class mappings
│ ├── agent_id_classes.npy
│ ├── agent_ip_classes.npy
│ ├── agent_name_classes.npy
│ ├── attack_chain_classes.npy
│ ├── eventdata_image_classes.npy
│ ├── mitre_id_classes.npy
│ └── rule_id_classes.npy
├── data/
│ └── generated_wazuh_logs.jsonl     # Raw logs from Wazuh API (JSONL)
├── model/
│ ├── attack_chain_vectors.npy       # Precomputed attack chain embeddings
│ ├── build_model.py                 # Train LSTM model, output metrics
│ └── lstm_attack_chain_model.keras  # Saved trained model
├── preprocessing/
│ └── prepare_dataset.py             # Label encoding, padding, slicing
├── utils/
│ └── wazuh_api.py                   # Script to fetch and parse Wazuh API logs
├── Realtime_detection.py            # Real-time inference & similarity computation
├── app.py                           # Live dashboard via Dash framework
├── encoders.pkl                     # Serialized LabelEncoders
├── model_info.txt                   # Model hyperparameters and info
├── output_confusion_matrix.png      # Confusion matrix visualization
├── output_roc_auc.png               # ROC curve visualization
├── output_training_history.png      # Accuracy/loss training curves
└── requirements.txt                 # Python dependency list










## 📌 Project Features

- 🔍 **LSTM-based sequential behavior classification**
- 🛡️ **Wazuh integration** for log collection from ICS environments
- 📊 **Real-time similarity scoring** between observed events and trained attack vectors
- 📉 **Model performance visualization** (Confusion Matrix, ROC/AUC, Training History)
- ⚙️ **Dash-powered dashboard** for live monitoring
- 🔁 **Modular design** for easy reuse of preprocessing, model, and API logic
