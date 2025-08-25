# Realtime_Transformer_Chain_Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
## Project Overview

This project leverages a Transformer-based model to detect and classify three types of cyber attack chains by analyzing event logs collected from Wazuh. It identifies abnormal sequences based on similarity scoring with pre-trained attack patterns and visualizes predictions through a real-time Dash dashboard.

## Project Features

- **Transformer-based sequential behavior classification**
- **Wazuh integration** for real-time security event collection
- **Real-time similarity scoring** between new log sequences and learned attack behaviors
- **Model performance visualization** Confusion Matrix, ROC Curve, AUC, Training History
- **Live dashboard** for monitoring detection results and model confidence
- **Modular design** for preprocessing, training, inference, and visualization


## Project Structure

```bash
Realtime_Transformer_Chain_Detection/
├── Model/                        # Model evaluation outputs
│ ├── Report/
│ │ ├── prediction_confidence_distribution_Chain0.png
│ │ ├── prediction_confidence_distribution_Chain1.png
│ │ ├── prediction_confidence_distribution_Chain2.png
│ │ ├── prediction_results_Chain0.csv
│ │ ├── prediction_results_Chain1.csv
│ │ └── prediction_results_Chain2.csv
│ ├── improved_preprocessors.pkl    # Encoders and transformers
│ ├── improved_transformer_model.keras  # # Trained Transformer model
│ ├── inference.py    # Real-time prediction & scoring
│ ├── model_build.py  # Train Transformer model, output metrics
│── api/
│   ├── inference.py       # 模型推論
│   ├── train.py           # retrain 機制
│   ├── auto_label.py      # pseudo-label + K-means
│   ├── scheduler.py       # 自動 retrain 排程
│   └── app.py             # FastAPI 主服務
├── data/
│ ├── attack_chain_0.jsonl     # Training logs for Chain 0
│ ├── attack_chain_1.jsonl     # Training logs for Chain 1
│ ├── attack_chain_2.jsonl     # Training logs for Chain 2
│ └── new_attack_data.jsonl    # New log data for evaluation
│   ├── auto_labeled_logs.json   # 自動標記 log
│   └── uncertain_logs.json      # 低信心 log
├── docker/
│ ├── .dockerignore
│ ├── Dockerfile
│ └── docker-compose.yml
├── utils/
│ ├── .env.local
│ ├── clear_log.sh
│ └── wazuh_api.py                  # Wazuh API integration script
├── Model_confusion_matrix.png      # Confusion matrix plot
├── Model_roc_auc.png               # ROC curve and AUC plot
├── Model_training_history.png      # Training loss/accuracy
├── app.py                          # Dash-based dashboard
└── requirements.txt                # Python dependency list
```

## Setup Instructions
### Environment
- Operating System: Ubuntu 22.04 (recommended)
- RAM: 12GB

### 1. Clone the Repository

```bash
git clone https://github.com/yuxiutan/Realtime_Transformer_Chain_Detection.git
cd Realtime_Transformer_Chain_Detection
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Use

### Step 1: Prepare Dataset and Train the Model

```bash
python Model/model_build.py
```

### Step 2: Run Real-time Detection

```bash
python inference.py
```

### Step 4: Launch the Dashboard

```bash
python app.py
```

### Step 5 (Optional) : Automated scheduling

```bash
sudo apt install jq

chmod +x /home/youruser/Realtime_Transformer_Chain_Detection/utils/clear_log.sh
chmod +x /home/youruser/Realtime_Transformer_Chain_Detection/utils/wazuh_api.sh

chown youruser:youruser /home/youruser/Realtime_Transformer_Chain_Detection/data
chmod u+w /home/youruser/Realtime_Transformer_Chain_Detection/data

crontab -e

0 0 * * * /bin/bash /home/youruser/Realtime_Transformer_Chain_Detection/utils/clear_log.sh
*/5 * * * * /bin/bash /home/youruser/Realtime_Transformer_Chain_Detection/utils/wazuh_api.sh

```
## Docker Quick Deployment
### Prerequisites
- Docker
- Docker Compose

### 1. Clone the Repository

```bash
git clone https://github.com/yuxiutan/Realtime_Transformer_Chain_Detection.git
cd Realtime_Transformer_Chain_Detection
```

### 2. Build the Docker Image

```bash
docker-compose build
```

### 3. Run the Docker Containers

```bash
docker-compose up -d
```

### 4. Access the Dashboard
- http://localhost:8050

### Stop the Docker Containers

```bash
docker-compose down
```

### Rebuild Docker Image

```bash
docker-compose build --no-cache
```

## Visual Outputs
- Model_confusion_matrix.png – Model confusion matrix
- Model_roc_auc.png – ROC curve and AUC
- Model_training_history.png – Training accuracy and loss trends

## Model Highlights
- Transformer Encoder for capturing complex temporal behavior
- Sequence length: 10 consecutive logs as model input
- Input features: Encoded fields like agent.ip, agent.name, rule.id, etc.
- Output: Softmax-based attack chain prediction
- Similarity scoring with class vectors to detect potential anomalies

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- Special thanks to the experts who provided technical support and advice.
