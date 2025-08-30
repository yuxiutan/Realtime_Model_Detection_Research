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
# 使用 try-except 處理可能的 ValueError，例如非標準 IP 格式
def get_last_octet(ip):
    try:
        return int(ip.split('.')[-1])
    except (ValueError, IndexError):
        # 如果格式不正確，可以返回一個預設值，例如 0 或 NaN，或者跳過該行
        print(f"Warning: Could not parse IP address: {ip}. Returning 0 for agent_ip.")
        return 0 # 或 np.nan，根據您的需求選擇

df['agent_ip'] = df['agent.ip'].apply(get_last_octet)

# Extract numerical part from agent.id (e.g., 'agent-007' → 7)
# 確保 agent.id 始終是字串，以防非預期格式
def get_agent_id_num(aid):
    try:
        if isinstance(aid, (int, float)): # 如果已经是数字，直接返回
            return int(aid)
        return int(str(aid).split('-')[-1])
    except (ValueError, IndexError):
        print(f"Warning: Could not parse agent.id: {aid}. Returning 0 for agent_id.")
        return 0

df['agent_id'] = df['agent.id'].apply(get_agent_id_num)

df['rule_id'] = df['rule.id']

# === IMPORTANT FIX FOR mitre_id ===
# PROBLEM: df['rule.mitre.id'] contains a mix of strings and lists of strings.
# LabelEncoder requires uniform input (all strings or all numbers).
# SOLUTION: Convert all 'rule.mitre.id' entries into a single, comma-separated string.
def standardize_mitre_id(mitre_id_entry):
    if isinstance(mitre_id_entry, list):
        # If it's a list, join elements into a comma-separated string
        return ",".join(mitre_id_entry)
    elif isinstance(mitre_id_entry, str):
        # If it's already a string, return as is
        return mitre_id_entry
    else:
        # Handle unexpected types by converting them to string
        print(f"Warning: Unexpected type for mitre_id: {type(mitre_id_entry)} - {mitre_id_entry}. Converting to string.")
        return str(mitre_id_entry)

df['mitre_id'] = df['rule.mitre.id'].apply(standardize_mitre_id)

df['eventdata_image'] = df['full_log'] # Renaming for consistency with features list

# === Label Encoding for Categorical Features ===
# 確保這些欄位在應用 LabelEncoder 之前是字串類型
df['agent.name'] = df['agent.name'].astype(str)
df['full_log'] = df['full_log'].astype(str)
df['attack_chain'] = df['attack_chain'].astype(str)


le_agent_name = LabelEncoder().fit(df['agent.name'])
le_event_image = LabelEncoder().fit(df['eventdata_image'])
le_mitre_id = LabelEncoder().fit(df['mitre_id']) # Now df['mitre_id'] is uniformly strings
le_attack_chain = LabelEncoder().fit(df['attack_chain'])

df['agent_name_encoded'] = le_agent_name.transform(df['agent.name'])
df['eventdata_image_encoded'] = le_event_image.transform(df['eventdata_image'])
df['mitre_id_encoded'] = le_mitre_id.transform(df['mitre_id'])
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
for feature_name, encoder in encoders.items():
    class_file = f"classes/{feature_name}_classes.npy"
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
# 使用編碼後的欄位作為特徵
features = ['agent_ip', 'agent_name_encoded', 'agent_id', 'eventdata_image_encoded', 'rule_id', 'mitre_id_encoded']
labels = df['attack_chain_encoded'].values

# === Generate LSTM-Compatible Sequences ===
SEQ_LEN = 10
X, y = [], []

# 确保 DataFrame 的长度至少是 SEQ_LEN，否则 range(len(df) - SEQ_LEN) 可能为负或 0
if len(df) < SEQ_LEN:
    print(f"Warning: DataFrame has fewer rows ({len(df)}) than SEQ_LEN ({SEQ_LEN}). No sequences will be generated.")
else:
    for i in range(len(df) - SEQ_LEN + 1): # 修正循环范围以包含所有可能的序列
        seq_x = df[features].iloc[i:i+SEQ_LEN].values
        # 对于链检测，y通常是序列的下一个状态或整个序列的标签
        # 这里使用序列结束时的标签作为该序列的标签
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
