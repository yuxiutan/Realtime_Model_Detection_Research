import json
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load JSONL data
with open("data/generated_wazuh_logs.jsonl", "r") as f:
    logs = [json.loads(line) for line in f]
df = pd.DataFrame(logs)

# === Field Preprocessing ===
# Extract the last octet of agent.ip for numerical processing
df['agent_ip'] = df['agent.ip'].apply(lambda ip: int(ip.split('.')[-1]))
# Extract numerical part from agent.id (e.g., 'agent-007' → 7)
df['agent_id'] = df['agent.id'].apply(lambda aid: int(aid.split('-')[-1]))
df['rule_id'] = df['rule.id']
df['mitre_id'] = df['rule.mitre.id']
df['eventdata_image'] = df['full_log']

# === Label Encoding for Categorical Features ===
le_agent_name = LabelEncoder().fit(df['agent.name'])
le_event_image = LabelEncoder().fit(df['eventdata_image'])
le_mitre_id = LabelEncoder().fit(df['mitre_id'])
le_attack_chain = LabelEncoder().fit(df['attack_chain'])

df['agent_name'] = le_agent_name.transform(df['agent.name'])
df['eventdata_image'] = le_event_image.transform(df['eventdata_image'])
df['mitre_id'] = le_mitre_id.transform(df['mitre_id'])
df['attack_chain_encoded'] = le_attack_chain.transform(df['attack_chain'])

# Save label encoders
encoders = {
    'agent_name': le_agent_name,
    'eventdata_image': le_event_image,
    'mitre_id': le_mitre_id,
    'attack_chain': le_attack_chain
}
joblib.dump(encoders, "encoders.pkl")
print("LabelEncoders saved to encoders.pkl")

# === Save All Classes for Each Feature ===
# Create the classes directory
os.makedirs("classes", exist_ok=True)

# Save encoded class labels for categorical features
for feature, encoder in encoders.items():
    class_file = f"classes/{feature}_classes.npy"
    np.save(class_file, encoder.classes_)
    print(f"Saved: {class_file}")

# Save unique values for numeric features
numeric_features = ['agent_ip', 'agent_id', 'rule_id']
for feature in numeric_features:
    unique_values = np.sort(df[feature].unique())
    class_file = f"classes/{feature}_classes.npy"
    np.save(class_file, unique_values)
    print(f"Saved numeric feature: {class_file}")

# === Define Feature and Label Columns ===
features = ['agent_ip', 'agent_name', 'agent_id', 'eventdata_image', 'rule_id', 'mitre_id']
labels = df['attack_chain_encoded'].values

# === Generate LSTM-Compatible Sequences ===
SEQ_LEN = 10
X, y = [], []
for i in range(len(df) - SEQ_LEN):
    seq_x = df[features].iloc[i:i+SEQ_LEN].values
    seq_y = labels[i + SEQ_LEN - 1]
    X.append(seq_x)
    y.append(seq_y)

X = np.array(X)
y = np.array(y)

# === Split Dataset into Training and Testing Sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Save Final Dataset to Disk ===
np.savez("data/lstm_dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print(f"A total of {len(X)} sequences were generated and saved to data/lstm_dataset.npz")
