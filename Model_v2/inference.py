import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import ipaddress
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model and preprocessors
def load_model_and_preprocessors(model_path='improved_chain_aware_lstm_model.keras', 
                                preprocessor_path='improved_preprocessors.pkl'):
    """Load the trained model and preprocessors"""
    print("Loading model and preprocessors...")
    try:
        model = tf.keras.models.load_model(model_path)
        with open(preprocessor_path, 'rb') as f:
            preprocessors = pickle.load(f)
        tokenizer = preprocessors['tokenizer']
        scaler = preprocessors['scaler']
        print("Model and preprocessors loaded successfully!")
        return model, tokenizer, scaler
    except Exception as e:
        raise ValueError(f"Failed to load model or preprocessors: {e}")

# Load new data
def load_new_jsonl_file(file_path, chain_instance_id="test_chain"):
    """Load a new JSONL file and assign a chain_instance_id"""
    print(f"Loading new data file {file_path}...")
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                record['chain_instance_id'] = chain_instance_id  # Assign a unique chain_instance_id to new data
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                continue
    
    if not data:
        raise ValueError("No valid data in the new data file")
    
    df = pd.DataFrame(data)
    print(f"Successfully loaded {len(df)} logs")
    return df

# Preprocess new data
def preprocess_new_data(df, tokenizer, scaler, max_sequence_length=40):
    """Preprocess new data consistently with training"""
    print("Starting preprocessing of new data...")
    
    # Check required columns
    required_columns = ['@timestamp', 'agent.ip', 'agent.name', 'agent.id', 'rule.id', 'rule.mitre.id', 'full_log']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing columns in new data: {missing}")
    
    # Sort by chain_instance_id and timestamp
    df = df.sort_values(['chain_instance_id', '@timestamp']).reset_index(drop=True)
    
    # Handle @timestamp
    df['@timestamp'] = pd.to_numeric(df['@timestamp'], errors='coerce')
    df['@timestamp'] = df['@timestamp'].fillna(df['@timestamp'].median())
    
    # Extract time features
    df['hour'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.dayofweek
    
    # Normalize timestamp
    timestamp_std = df['@timestamp'].std()
    if timestamp_std == 0 or np.isnan(timestamp_std):
        df['@timestamp_normalized'] = df['@timestamp']
    else:
        df['@timestamp_normalized'] = (df['@timestamp'] - df['@timestamp'].mean()) / timestamp_std
    
    # Calculate time difference
    df['time_diff'] = df.groupby('chain_instance_id')['@timestamp'].diff().fillna(0)
    time_diff_std = df['time_diff'].std()
    if time_diff_std == 0 or np.isnan(time_diff_std):
        df['time_diff_normalized'] = df['time_diff']
    else:
        df['time_diff_normalized'] = (df['time_diff'] - df['time_diff'].mean()) / time_diff_std
    
    # Convert IP address to integer and normalize
    def ip_to_int(ip):
        try:
            return int(ipaddress.IPv4Address(ip))
        except:
            return 0
    
    df['agent.ip_int'] = df['agent.ip'].apply(ip_to_int)
    ip_std = df['agent.ip_int'].std()
    if ip_std == 0 or np.isnan(ip_std):
        df['agent.ip_normalized'] = df['agent.ip_int']
    else:
        df['agent.ip_normalized'] = (df['agent.ip_int'] - df['agent.ip_int'].mean()) / ip_std
    
    # Calculate log length feature
    df['log_length'] = df['full_log'].astype(str).str.len()
    df['log_length_normalized'] = (df['log_length'] - df['log_length'].mean()) / df['log_length'].std()
    
    # Calculate position feature within attack chain
    df['position_in_chain'] = df.groupby('chain_instance_id').cumcount()
    df['chain_total_length'] = df.groupby('chain_instance_id')['chain_instance_id'].transform('count')
    df['position_ratio'] = df['position_in_chain'] / df['chain_total_length']
    
    # Label Encoding (using the encoder from training, assuming it was saved)
    # Assume new data category values have been seen in training data
    # Handle unknown categories if present (e.g., set to default value)
    df['agent.name_encoded'] = df['agent.name'].astype(str).map(
        lambda x: tokenizer.word_index.get(x, 0) if hasattr(tokenizer, 'word_index') else 0
    )
    df['agent.id_encoded'] = df['agent.id'].astype(str).map(
        lambda x: tokenizer.word_index.get(x, 0) if hasattr(tokenizer, 'word_index') else 0
    )
    df['rule.id_encoded'] = df['rule.id'].astype(str).map(
        lambda x: tokenizer.word_index.get(x, 0) if hasattr(tokenizer, 'word_index') else 0
    )
    df['rule.mitre.id_encoded'] = df['rule.mitre.id'].astype(str).map(
        lambda x: tokenizer.word_index.get(x, 0) if hasattr(tokenizer, 'word_index') else 0
    )
    
    # Tokenize and sequence full_log
    sequences = tokenizer.texts_to_sequences(df['full_log'].astype(str))
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Extract static features
    static_features = df[['@timestamp_normalized', 'time_diff_normalized', 'agent.ip_normalized', 
                         'agent.name_encoded', 'agent.id_encoded', 'rule.id_encoded', 'rule.mitre.id_encoded',
                         'hour', 'day_of_week', 'log_length_normalized', 'position_ratio']].values
    
    # Handle NaN and infinite values
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Standardize using the scaler from training
    static_features = scaler.transform(static_features)
    
    return padded_sequences, static_features, df

# Perform model inference
def perform_inference(model, X_seq, X_static, df, output_file='inference_results.csv'):
    """Perform inference on new data and save results"""
    print("Starting model inference...")
    
    # Make predictions
    predictions = model.predict({'sequence_input': X_seq, 'static_input': X_static}, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    prediction_confidences = np.max(predictions, axis=1)
    
    # Compile results
    results = pd.DataFrame({
        'chain_instance_id': df['chain_instance_id'],
        '@timestamp': df['@timestamp'],
        'full_log': df['full_log'],
        'predicted_chain': predicted_classes,
        'confidence': prediction_confidences
    })
    
    # Add probabilities for each class
    for i in range(3):
        results[f'prob_chain_{i}'] = predictions[:, i]
    
    # Print first few prediction results
    print(f"\nFirst {min(10, len(results))} prediction results:")
    for idx, row in results.head_GB(10).iterrows():
        print(f"Sample {idx+1}:")
        print(f"  Chain Instance ID: {row['chain_instance_id']}")
        print(f"  Timestamp: {row['@timestamp']}")
        print(f"  Log: {row['full_log'][:100]}...")  # Display only first 100 characters
        print(f"  Predicted Attack Chain: {row['predicted_chain']}")
        print(f"  Confidence: {row['confidence']:.3f}")
        print(f"  Probabilities: Chain 0={row['prob_chain_0']:.3f}, "
              f"Chain 1={row['prob_chain_1']:.3f}, Chain 2={row['prob_chain_2']:.3f}")
        print("-" * 50)
    
    # Summarize predictions by chain_instance_id
    chain_summary = results.groupby(['chain_instance_id', 'predicted_chain']).size().unstack(fill_value=0)
    print("\nPrediction distribution for each attack chain:")
    print(chain_summary)
    
    # Visualize confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results['confidence'], bins=30, kde=True)
    plt.title('Prediction Confidence Distribution', fontsize=14)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("Confidence distribution plot saved as 'confidence_distribution.png'")
    plt.show()
    
    # Save results to CSV
    results.to_csv(output_file, index=False)
    print(f"Inference results saved as '{output_file}'")
    
    return results, chain_summary

# Main program - inference test
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Assume new data file path
    new_data_file = 'new_attack_data.jsonl'  # Replace with your new data file path
    
    # Step 1: Load model and preprocessors
    model, tokenizer, scaler = load_model_and_preprocessors()
    
    # Step 2: Load new data
    new_df = load_new_jsonl_file(new_data_file, chain_instance_id="test_chain_001")
    
    # Step 3: Preprocess new data
    X_seq_new, X_static_new, processed_df = preprocess_new_data(
        new_df, tokenizer, scaler, max_sequence_length=40
    )
    
    # Step 4: Perform inference and save results
    results, chain_summary = perform_inference(
        model, X_seq_new, X_static_new, processed_df, output_file='inference_results.csv'
    )
    
    print("\nInference completed!")
    print("="*50)
    print("Generated files:")
    print("1. inference_results.csv - Detailed inference results")
    print("2. confidence_distribution.png - Prediction confidence distribution plot")
    print("="*50)
