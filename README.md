# Realtime_LSTM_Chain_Detection
# 🔐 LSTM-Based Behavior Detection System using Wazuh Logs

A real-time industrial cybersecurity behavior detection system that leverages **Wazuh logs**, **LSTM deep learning**, and **similarity-based attack chain recognition**, built with **Python**, **Keras**, and **Dash**.

## 📌 Project Overview

This project aims to detect abnormal behaviors in industrial OT environments by training an LSTM model with logs collected from **Wazuh** via API. It identifies potential multi-step attack chains and triggers alerts based on similarity scores. A real-time monitoring dashboard is provided using **Dash**.

---

## 📂 Project Structure

```bash
LSTM-Behavior-Detection/
├── app.py                     # Dash-based real-time dashboard
├── build_model.py             # Train and evaluate the LSTM model
├── Realtime_detection.py      # Run real-time inference and compute similarity
├── requirements.txt           # List of dependencies
├── generated_wazuh_logs.jsonl # Raw training dataset from Wazuh API
├── model/
│   ├── evaluate.py            # Metrics and confusion matrix
│   ├── lstm_model.py          # LSTM model architecture
│   └── test_model.py          # Testing and performance verification
├── preprocessing/
│   └── prepare_dataset.py     # Label encoding, padding, slicing
└── tools/
    ├── read_device.py         # (optional) Read data from Modbus device
    ├── write_device.py        # (optional) Write to device
    └── modbus_tcpdump_log.py  # (optional) Log generator from Modbus traffic
