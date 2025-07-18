import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ipaddress
import os
import pickle
from scipy.stats import gaussian_kde
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def ip_to_int(ip):
    try:
        return int(ipaddress.IPv4Address(ip))
    except:
        return 0

def load_jsonl_data(file_path):
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on line: {e}")
                continue
    
    if not data:
        raise ValueError("No data successfully loaded")
    
    df = pd.DataFrame(data)
    print(f"Successfully loaded {len(df)} log entries")
    return df

def preprocess_new_data(df, tokenizer, scaler, vectorizer, max_sequence_length=60):
    df = df.copy()
    
    # Ensure required columns exist
    required_columns = ['@timestamp', 'agent.ip', 'agent.name', 'agent.id', 'rule.id', 'rule.mitre.id', 'full_log']
    for col in required_columns:
        if col not in df.columns:
            df[col] = '' if col == 'full_log' else 0
    
    # Data validation
    if df['@timestamp'].isna().all():
        raise ValueError("All @timestamp values are NaN")
    if df['full_log'].isna().all() or df['full_log'].eq('').all():
        raise ValueError("All full_log values are NaN or empty strings")
    
    # Time feature processing
    df['@timestamp'] = pd.to_numeric(df['@timestamp'], errors='coerce')
    df['@timestamp'] = df['@timestamp'].fillna(df['@timestamp'].median())
    
    df['hour'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.dayofweek
    timestamp_std = df['@timestamp'].std()
    if timestamp_std == 0 or np.isnan(timestamp_std):
        df['@timestamp_normalized'] = df['@timestamp']
    else:
        df['@timestamp_normalized'] = (df['@timestamp'] - df['@timestamp'].mean()) / timestamp_std
    
    df['time_diff'] = 0
    df['time_diff_normalized'] = 0
    
    # IP address conversion
    df['agent.ip_int'] = df['agent.ip'].apply(ip_to_int)
    ip_std = df['agent.ip_int'].std()
    if ip_std == 0 or np.isnan(ip_std):
        df['agent.ip_normalized'] = df['agent.ip_int']
    else:
        df['agent.ip_normalized'] = (df['agent.ip_int'] - df['agent.ip_int'].mean()) / ip_std
    
    # Log length
    df['log_length'] = df['full_log'].astype(str).str.len()
    log_length_std = df['log_length'].std()
    if log_length_std == 0 or np.isnan(log_length_std):
        df['log_length_normalized'] = df['log_length']
    else:
        df['log_length_normalized'] = (df['log_length'] - df['log_length'].mean()) / log_length_std
    
    df['position_in_chain'] = 0
    df['chain_total_length'] = 1
    df['position_ratio'] = 0
    
    df['rule.id'] = df['rule.id'].astype(str)
    df['prev_rule_id'] = 'NONE'
    df['next_rule_id'] = 'NONE'
    
    # MITRE ATT&CK features
    df['mitre_tactic'] = df['rule.mitre.id'].apply(lambda x: str(x).split('.')[0] if isinstance(x, (str, int, float)) else 'UNKNOWN')
    df['mitre_technique'] = df['rule.mitre.id'].apply(lambda x: str(x) if isinstance(x, (str, int, float)) else 'UNKNOWN')
    
    df['agent.name'] = df['agent.name'].astype(str)
    df['agent.id'] = df['agent.id'].astype(str)
    
    # Skip LabelEncoder, use fixed values
    df['agent.name_encoded'] = 0
    df['agent.id_encoded'] = 0
    df['rule.id_encoded'] = 0
    df['mitre_tactic_encoded'] = 0
    df['mitre_technique_encoded'] = 0
    df['prev_rule_id_encoded'] = 0
    df['next_rule_id_encoded'] = 0
    
    # N-gram features
    ngram_features = vectorizer.transform(df['full_log'].astype(str)).toarray()
    ngram_columns = vectorizer.get_feature_names_out()
    df_ngram = pd.DataFrame(ngram_features, columns=[f'ngram_{col}' for col in ngram_columns])
    df = pd.concat([df, df_ngram], axis=1)
    
    # Character-level tokenization
    sequences = tokenizer.texts_to_sequences(df['full_log'].astype(str))
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Static features
    static_features_columns = [
        '@timestamp_normalized', 'time_diff_normalized', 'agent.ip_normalized',
        'agent.name_encoded', 'agent.id_encoded', 'rule.id_encoded',
        'mitre_tactic_encoded', 'mitre_technique_encoded',
        'prev_rule_id_encoded', 'next_rule_id_encoded',
        'hour', 'day_of_week', 'log_length_normalized', 'position_ratio'
    ] + [f'ngram_{col}' for col in ngram_columns]
    
    static_features = df[static_features_columns].values
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=1.0, neginf=-1.0)
    static_features = scaler.transform(static_features)
    
    return padded_sequences, static_features, df

def send_discord_alert(webhook_url, message):
    data = {
        "content": message,
        "username": "Security Alert Bot"
    }
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Discord alert sent successfully")
        else:
            print(f"Discord alert failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send Discord alert: {e}")

def send_wazuh_alert(wazuh_api_url, username, password, alert_message, agent_id="000", rule_id="100005", level=12):
    try:
        # Wazuh API authentication
        auth_url = f"{wazuh_api_url}/security/user/authenticate"
        auth_response = requests.get(auth_url, auth=(username, password), verify=False)
        auth_response.raise_for_status()
        token = auth_response.json()['data']['token']

        # Construct alert log string with required metadata
        log_message = (
            f"[Custom Alert] rule_id={rule_id} level={level} agent_id={agent_id} "
            f"description=Custom high-confidence attack chain alert {alert_message}"
        )

        # Build alert data for Wazuh API /events endpoint
        alert_data = {
            "events": [log_message]
        }

        # Send alert to Wazuh API
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        log_url = f"{wazuh_api_url}/events"
        response = requests.post(log_url, headers=headers, json=alert_data, verify=False)
        
        if response.status_code in [200, 201]:
            print("Wazuh alert sent successfully")
        else:
            print(f"Failed to send Wazuh alert: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"Failed to send Wazuh alert: {e}")

def predict_new_data(df, transformer_model, tokenizer, scaler, vectorizer, max_sequence_length=60):
    X_seq, X_static, processed_df = preprocess_new_data(df, tokenizer, scaler, vectorizer, max_sequence_length)
    
    transformer_pred = transformer_model.predict({'sequence_input': X_seq, 'static_input': X_static})
    
    ensemble_pred = transformer_pred
    predicted_classes = np.argmax(ensemble_pred, axis=1)
    prediction_confidences = np.max(ensemble_pred, axis=1)
    
    results = pd.DataFrame({
        'full_log': df['full_log'],
        'predicted_chain': predicted_classes,
        'confidence': prediction_confidences
    })
    
    for i in range(3):
        results[f'chain_{i}_probability'] = ensemble_pred[:, i]
    
    return results, processed_df

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * y_true * tf.pow(1. - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fn

if __name__ == "__main__":
    # Configuration (loaded from .env file)
    WAZUH_API_URL = os.getenv("WAZUH_API_URL")
    WAZUH_API_USERNAME = os.getenv("WAZUH_API_USERNAME")
    WAZUH_API_PASSWORD = os.getenv("WAZUH_API_PASSWORD")
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

    # Check for missing required environment variables
    if not all([WAZUH_API_URL, WAZUH_API_USERNAME, WAZUH_API_PASSWORD, DISCORD_WEBHOOK_URL]):
        raise ValueError("One or more required environment variables are missing (WAZUH_API_URL, WAZUH_API_USERNAME, WAZUH_API_PASSWORD, DISCORD_WEBHOOK_URL)")

    try:
        transformer_model = tf.keras.models.load_model('improved_transformer_model.keras', 
                                                     custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)})
        print("Transformer model loaded successfully")
    except Exception as e:
        print(f"Failed to load Transformer model: {e}")
        exit(1)
    
    try:
        with open('improved_preprocessors.pkl', 'rb') as f:
            preprocessors = pickle.load(f)
            tokenizer = preprocessors['tokenizer']
            scaler = preprocessors['scaler']
            vectorizer = preprocessors['vectorizer']
        print("Preprocessors loaded successfully")
    except Exception as e:
        print(f"Failed to load preprocessors: {e}")
        exit(1)
    
    file_path = '/home/danish/Realtime_Model_Detection_Research/data/new_attack_data.jsonl'
    try:
        new_data = load_jsonl_data(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        exit(1)
    
    try:
        results, processed_df = predict_new_data(
            new_data,
            transformer_model,
            tokenizer,
            scaler,
            vectorizer
        )
        
        print("\nInference results (first 10 rows, see CSV file for complete results):")
        print(results[['full_log', 'predicted_chain', 'confidence', 
                      'chain_0_probability', 'chain_1_probability', 'chain_2_probability']].head(10))
        
        # List all unique agent.name and agent.ip
        unique_agent_names = new_data['agent.name'].dropna().unique().tolist()
        unique_agent_ips = new_data['agent.ip'].dropna().unique().tolist()
        
        print("\nDetected unique Agent names:")
        print(", ".join(str(name) for name in unique_agent_names) if unique_agent_names else "None")
        print("\nDetected unique Agent IP addresses:")
        print(", ".join(str(ip) for ip in unique_agent_ips) if unique_agent_ips else "None")
        
        # Find the most frequent attack chain category and its average confidence
        if not results['predicted_chain'].empty:
            most_frequent_chain = results['predicted_chain'].value_counts().idxmax()
            avg_confidence = results['confidence'].mean()
            print(f"\nMost frequent attack chain category: {most_frequent_chain}")
            print(f"Average confidence: {avg_confidence:.4f}")
            
            # Check if average confidence is greater than 0.7 and send alerts
            if avg_confidence > 0.7:
                alert_message = (
                    f"⚠️ High Confidence Attack Chain Detected ⚠️\n"
                    f"Most frequent attack chain category: {most_frequent_chain}\n"
                    f"Average confidence: {avg_confidence:.4f}\n"
                    f"Number of detected logs: {len(results)}\n"
                    f"Detected unique Agent names: {', '.join(str(name) for name in unique_agent_names) if unique_agent_names else 'None'}\n"
                    f"Detected unique Agent IP addresses: {', '.join(str(ip) for ip in unique_agent_ips) if unique_agent_ips else 'None'}\n"
                    f"Time: {pd.Timestamp.now()}"
                )
                # Send Discord alert
                send_discord_alert(DISCORD_WEBHOOK_URL, alert_message)
                # Send Wazuh alert
                agent_id = new_data['agent.id'].dropna().iloc[0] if 'agent.id' in new_data and not new_data['agent.id'].dropna().empty else "000"
                send_wazuh_alert(
                    wazuh_api_url=WAZUH_API_URL,
                    username=WAZUH_API_USERNAME,
                    password=WAZUH_API_PASSWORD,
                    alert_message=alert_message,
                    agent_id=agent_id,
                    rule_id="100005",
                    level=12
                )
        else:
            print("\nNo valid attack chain prediction results")
        
        results.to_csv('prediction_results.csv', index=False)
        print("\nInference results saved as 'prediction_results.csv'")

        # Generate prediction confidence distribution plot
        plt.figure(figsize=(10, 6))
        confidence_values = results['confidence']
        
        # Plot histogram
        plt.hist(confidence_values, bins=20, color='skyblue', alpha=0.7, label='Confidence Distribution')
        
        # Calculate and plot KDE curve
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
        
        # Save the plot
        plt.savefig('prediction_confidence_distribution.png', dpi=300, bbox_inches='tight')
        print("Prediction confidence distribution plot saved as 'prediction_confidence_distribution.png'")
        plt.close()

    except Exception as e:
        print(f"Inference process failed: {e}")
        exit(1)
