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
                print(f"JSON parsing error at line: {e}")
                continue
    
    if not data:
        raise ValueError("No data loaded successfully")
    
    df = pd.DataFrame(data)
    print(f"Successfully loaded {len(df)} logs")
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
    try:
        transformer_model = tf.keras.models.load_model('Model/improved_transformer_model.keras', 
                                                     custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)})
        print("Transformer model loaded successfully")
    except Exception as e:
        print(f"Failed to load Transformer model: {e}")
        exit(1)
    
    try:
        with open('Model/improved_preprocessors.pkl', 'rb') as f:
            preprocessors = pickle.load(f)
            tokenizer = preprocessors['tokenizer']
            scaler = preprocessors['scaler']
            vectorizer = preprocessors['vectorizer']
        print("Preprocessors loaded successfully")
    except Exception as e:
        print(f"Failed to load preprocessors: {e}")
        exit(1)
    
    file_path = 'data/new_attack_data.jsonl'
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
        
        print("\nInference results (first 10, see CSV file for full results):")
        print(results[['full_log', 'predicted_chain', 'confidence', 
                      'chain_0_probability', 'chain_1_probability', 'chain_2_probability']].head(10))
        
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
        kde_values = kde(x_range) * len(confidence_values) * (max(confidence_values) - min(confidence_values)) / 20  # Scale adjustment to match histogram
        plt.plot(x_range, kde_values, 'b-', lw=2, label='KDE')
        
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('prediction_confidence_distribution.png', dpi=300, bbox_inches='tight')
        print("Prediction confidence distribution plot saved as 'prediction_confidence_distribution.png'")
        plt.close()

    except Exception as e:
        print(f"Inference process failed: {e}")
        exit(1)
