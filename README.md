# Realtime_Transformer_Chain_Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
## 📌 Project Overview

This project leverages a Transformer-based model to detect and classify three types of cyber attack chains by analyzing event logs collected from Wazuh. It identifies abnormal sequences based on similarity scoring with pre-trained attack patterns and visualizes predictions through a real-time Dash dashboard.

## 📌 Project Features

- 🔍 **Transformer-based sequential behavior classification**
- 🛡️ **Wazuh integration** for real-time security event collection
- 📊 **Real-time similarity scoring** between new log sequences and learned attack behaviors
- 📉 **Model performance visualization** Confusion Matrix, ROC Curve, AUC, Training History
- ⚙️ **Live dashboard** for monitoring detection results and model confidence
- 🔁 **Modular design** for preprocessing, training, inference, and visualization


## 📂 Project Structure

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
│ ├── attack_chain_0.jsonl
│ ├── attack_chain_1.jsonl
│ ├── attack_chain_2.jsonl
│ ├── improved_preprocessors.pkl
│ ├── improved_transformer_model.keras  # Saved trained model
│ ├── inference.py
│ ├── model_build.py  # Train LSTM model, output metrics
│ └── new_attack_data.jsonl
├── utils/
│ └── wazuh_api.py                   # Script to fetch and parse Wazuh API logs
├── model_info.txt                   # Model hyperparameters and info
├── Model_confusion_matrix.png      # Confusion matrix visualization
├── Model_roc_auc.png               # ROC curve visualization
├── Model_training_history.png      # Accuracy/loss training curves
├── app.py                           # Live dashboard via Dash framework
└── requirements.txt                 # Python dependency list
```

## ⚙️ Setup Instructions
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

## 🚀 How to Use

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

## 📈 Visual Outputs
- Model_confusion_matrix.png – Model confusion matrix
- Model_roc_auc.png – ROC curve and AUC
- Model_training_history.png – Training accuracy and loss trends

## 🧠 Model Highlights
- Transformer Encoder for capturing complex temporal behavior
- Sequence length: 10 consecutive logs as model input
- Input features: Encoded fields like agent.ip, agent.name, rule.id, etc.
- Output: Softmax-based attack chain prediction
- Similarity scoring with class vectors to detect potential anomalies

## 📄 License
This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements
- Special thanks to the experts who provided technical support and advice.
