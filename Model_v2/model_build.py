import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ipaddress
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Check duplicate rate of full_log
def check_log_duplicates(df):
    log_counts = df['full_log'].value_counts()
    total_logs = len(df)
    duplicates = log_counts[log_counts > 1]
    duplicate_rate = len(duplicates) / total_logs if total_logs > 0 else 0
    print(f"Total logs: {total_logs}")
    print(f"Unique logs: {len(log_counts)}")
    print(f"Duplicate log rate: {duplicate_rate:.2%}")
    print(f"Top 5 most frequent logs:\n{log_counts.head()}")

# Improved data loading function
def load_jsonl_files_with_chain_id(file_paths):
    """Load data and assign a unique ID to each attack chain"""
    data = []
    for i, file_path in enumerate(file_paths):
        print(f"Loading {file_path}...")
        if not os.path.exists(file_path):
            print(f"  Warning: File {file_path} does not exist, skipping...")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                try:
                    record = json.loads(line.strip())
                    record['chain_label'] = i  # 0, 1, 2 correspond to attack_chain_0, 1, 2
                    record['is_attack'] = 1  # All samples are attacks
                    # Each file represents a complete attack chain, so all logs have the same chain_instance_id
                    record['chain_instance_id'] = f"chain_type_{i}_file_{os.path.basename(file_path)}"
                    data.append(record)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"  JSON parsing error at line {line_count + 1}: {e}")
                    continue
            
            print(f"  Loaded {line_count} logs")
    
    if not data:
        raise ValueError("No data loaded successfully, please check file paths and formats")
    
    return pd.DataFrame(data)

# Improved feature processing function - adding more feature engineering
def preprocess_data(df, max_sequence_length=40, vocab_size=3000):  # Increase sequence length and vocabulary size
    # Check required columns
    required_columns = ['@timestamp', 'agent.ip', 'agent.name', 'agent.id', 'rule.id', 'rule.mitre.id', 'full_log']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing columns in data: {missing}")
    
    # Check duplicate rate of full_log
    check_log_duplicates(df)
    
    # Handle @timestamp - avoid NaN and division by zero
    df['@timestamp'] = pd.to_numeric(df['@timestamp'], errors='coerce')
    df['@timestamp'] = df['@timestamp'].fillna(df['@timestamp'].median())
    
    # Sort by attack chain and timestamp
    df = df.sort_values(['chain_instance_id', '@timestamp']).reset_index(drop=True)
    
    # Calculate time features
    df['hour'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.dayofweek
    
    # Normalize timestamp
    timestamp_std = df['@timestamp'].std()
    if timestamp_std == 0 or np.isnan(timestamp_std):
        df['@timestamp_normalized'] = df['@timestamp']
    else:
        df['@timestamp_normalized'] = (df['@timestamp'] - df['@timestamp'].mean()) / timestamp_std
    
    # Calculate time difference within attack chain
    df['time_diff'] = df.groupby('chain_instance_id')['@timestamp'].diff().fillna(0)
    
    # Normalize time difference
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
    
    # Label encode string features
    le_name = LabelEncoder()
    le_id = LabelEncoder()
    le_rule = LabelEncoder()
    le_mitre = LabelEncoder()
    
    df['agent.name_encoded'] = le_name.fit_transform(df['agent.name'].astype(str))
    df['agent.id_encoded'] = le_id.fit_transform(df['agent.id'].astype(str))
    df['rule.id_encoded'] = le_rule.fit_transform(df['rule.id'].astype(str))
    df['rule.mitre.id_encoded'] = le_mitre.fit_transform(df['rule.mitre.id'].astype(str))
    
    # Tokenize and sequence full_log - use larger vocabulary size
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", char_level=False)
    tokenizer.fit_on_texts(df['full_log'].astype(str))
    sequences = tokenizer.texts_to_sequences(df['full_log'].astype(str))
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Expand static features (include new time and position features)
    static_features = df[['@timestamp_normalized', 'time_diff_normalized', 'agent.ip_normalized', 
                         'agent.name_encoded', 'agent.id_encoded', 'rule.id_encoded', 'rule.mitre.id_encoded',
                         'hour', 'day_of_week', 'log_length_normalized', 'position_ratio']].values
    
    # Handle NaN or inf
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Further standardize using StandardScaler
    scaler = StandardScaler()
    static_features = scaler.fit_transform(static_features)
    
    # Attack chain labels (one-hot encoding)
    chain_labels = df['chain_label'].values
    chain_labels_onehot = np.zeros((len(chain_labels), 3))
    for i, label in enumerate(chain_labels):
        chain_labels_onehot[i, label] = 1
    
    return padded_sequences, static_features, chain_labels_onehot, tokenizer, scaler

# Improved LSTM model - adding complexity and attention mechanism
def build_improved_lstm_model(vocab_size, max_sequence_length, static_feature_size, use_attention=True):
    # Sequence input (full_log)
    sequence_input = tf.keras.Input(shape=(max_sequence_length,), name='sequence_input')
    
    # Larger embedding dimension
    embedding = tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True)(sequence_input)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    
    # Multi-layer Bidirectional LSTM
    lstm1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(embedding)
    
    lstm2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True if use_attention else False, 
                           dropout=0.3, recurrent_dropout=0.2)
    )(lstm1)
    
    if use_attention:
        # Add attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(lstm2, lstm2)
        attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
        lstm_output = attention
    else:
        lstm_output = lstm2
    
    # Static feature input - deeper network
    static_input = tf.keras.Input(shape=(static_feature_size,), name='static_input')
    static_dense1 = tf.keras.layers.Dense(64, activation='relu')(static_input)
    static_dense1 = tf.keras.layers.BatchNormalization()(static_dense1)
    static_dense1 = tf.keras.layers.Dropout(0.3)(static_dense1)
    
    static_dense2 = tf.keras.layers.Dense(32, activation='relu')(static_dense1)
    static_dense2 = tf.keras.layers.BatchNormalization()(static_dense2)
    static_dense2 = tf.keras.layers.Dropout(0.2)(static_dense2)
    
    # Combine LSTM and static features
    combined = tf.keras.layers.Concatenate()([lstm_output, static_dense2])
    
    # Deeper classification network
    dense1 = tf.keras.layers.Dense(256, activation='relu')(combined)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.4)(dense1)
    
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    dense3 = tf.keras.layers.Dense(64, activation='relu')(dense2)
    dense3 = tf.keras.layers.BatchNormalization()(dense3)
    dense3 = tf.keras.layers.Dropout(0.2)(dense3)
    
    # Output layer
    chain_output = tf.keras.layers.Dense(3, activation='softmax', name='chain_output')(dense3)
    
    model = tf.keras.Model(inputs=[sequence_input, static_input], outputs=chain_output)
    
    # Use better optimizer settings
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,  # Slightly higher initial learning rate
        weight_decay=0.01,    # Weight decay
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def chain_aware_train_test_split(df, test_size=0.2, random_state=42):
    """Split data based on attack chain files, ensuring all logs from the same file are in the same set"""
    
    # Get all unique attack chain files
    unique_chains = df['chain_instance_id'].unique()
    
    print(f"Total {len(unique_chains)} attack chain files:")
    for chain_id in unique_chains:
        chain_data = df[df['chain_instance_id'] == chain_id]
        chain_label = chain_data['chain_label'].iloc[0]
        log_count = len(chain_data)
        print(f"  {chain_id}: Attack Chain {chain_label}, {log_count} logs")
    
    # Group files by attack chain label
    chain_groups = {}
    for chain_id in unique_chains:
        chain_data = df[df['chain_instance_id'] == chain_id]
        chain_label = chain_data['chain_label'].iloc[0]
        
        if chain_label not in chain_groups:
            chain_groups[chain_label] = []
        chain_groups[chain_label].append(chain_id)
    
    print(f"\nAttack chain type distribution:")
    for label, chains in chain_groups.items():
        print(f"  Attack Chain {label}: {len(chains)} files")
    
    # Stratified split strategy - ensure sufficient samples in test set
    train_indices = []
    test_indices = []
    
    np.random.seed(random_state)
    
    for label, chain_files in chain_groups.items():
        if len(chain_files) == 1:
            # If only one file for this type, use a more conservative split ratio
            chain_id = chain_files[0]
            chain_indices = df[df['chain_instance_id'] == chain_id].index.tolist()
            
            # Shuffle indices randomly
            np.random.shuffle(chain_indices)
            
            # Ensure at least 5 samples or 20% for test set, whichever is larger
            n_test = max(5, int(len(chain_indices) * test_size))
            n_test = min(n_test, len(chain_indices) - 10)  # Ensure enough training data
            n_train = len(chain_indices) - n_test
            
            test_indices.extend(chain_indices[:n_test])
            train_indices.extend(chain_indices[n_test:])
            
            print(f"    Splitting {chain_id}: {n_train} training logs, {n_test} test logs")
        else:
            # If multiple files, split by file
            chains = np.array(chain_files)
            np.random.shuffle(chains)
            
            n_test_files = max(1, int(len(chains) * test_size))
            n_train_files = len(chains) - n_test_files
            
            test_files = chains[:n_test_files]
            train_files = chains[n_test_files:]
            
            # Collect corresponding indices
            for chain_id in test_files:
                test_indices.extend(df[df['chain_instance_id'] == chain_id].index.tolist())
            
            for chain_id in train_files:
                train_indices.extend(df[df['chain_instance_id'] == chain_id].index.tolist())
            
            print(f"    Attack Chain {label}: {n_train_files} training files, {n_test_files} test files")
    
    # Create train and test sets
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    print(f"\nFinal split results:")
    print(f"Training set: {len(train_df)} logs")
    print(f"Test set: {len(test_df)} logs")
    
    # Check class distribution after split
    print("\nTraining set class distribution:")
    train_counts = train_df['chain_label'].value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Attack Chain {label}: {count} logs")
    
    print("Test set class distribution:")
    test_counts = test_df['chain_label'].value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Attack Chain {label}: {count} logs")
    
    return train_df, test_df

def train_with_chain_aware_split(model, X_seq, X_static, y_chain, df, class_weight_dict, epochs=150, batch_size=32):
    """Train with attack chain-aware splitting"""
    
    # Perform chain-aware split
    train_df, test_df = chain_aware_train_test_split(df, test_size=0.2, random_state=42)
    
    # Get corresponding indices
    train_indices = train_df.index.tolist()
    test_indices = test_df.index.tolist()
    
    # Split features and labels
    X_seq_train = X_seq[train_indices]
    X_static_train = X_static[train_indices]
    y_train = y_chain[train_indices]
    
    X_seq_test = X_seq[test_indices]
    X_static_test = X_static[test_indices]
    y_test = y_chain[test_indices]
    
    # Split validation set from training set
    val_split = 0.15  # Reduce validation set ratio to increase training data
    n_val = int(len(X_seq_train) * val_split)
    
    # Randomly select validation set indices
    np.random.seed(42)
    val_indices = np.random.choice(len(X_seq_train), n_val, replace=False)
    train_indices_inner = np.setdiff1d(range(len(X_seq_train)), val_indices)
    
    X_seq_train_final = X_seq_train[train_indices_inner]
    X_static_train_final = X_static_train[train_indices_inner]
    y_train_final = y_train[train_indices_inner]
    
    X_seq_val = X_seq_train[val_indices]
    X_static_val = X_static_train[val_indices]
    y_val = y_train[val_indices]
    
    # Print final data distribution
    print("\nFinal data distribution:")
    print("Training set class distribution:")
    unique, counts = np.unique(y_train_final.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  Attack Chain {i}: {count} samples")
        
    print("Validation set class distribution:")
    unique, counts = np.unique(y_val.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  Attack Chain {i}: {count} samples")
        
    print("Test set class distribution:")
    unique, counts = np.unique(y_test.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  Attack Chain {i}: {count} samples")
    
    # Improved callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # Increase patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Add minimum improvement threshold
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3,  # More aggressive learning rate reduction
            patience=10, 
            min_lr=1e-8,
            verbose=1
        ),
        # Add learning rate scheduler
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.95 ** epoch), verbose=0
        )
    ]
    
    # Train model
    history = model.fit(
        {'sequence_input': X_seq_train_final, 'static_input': X_static_train_final},
        y_train_final,
        validation_data=({'sequence_input': X_seq_val, 'static_input': X_static_val}, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return history, X_seq_test, X_static_test, y_test

# Main program
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    file_paths = ['attack_chain_0.jsonl', 'attack_chain_1.jsonl', 'attack_chain_2.jsonl']
    df = load_jsonl_files_with_chain_id(file_paths)
    
    # Check attack chain file statistics
    print("Attack chain file statistics:")
    for label in [0, 1, 2]:
        if label in df['chain_label'].values:
            chain_files = df[df['chain_label'] == label]['chain_instance_id'].nunique()
            total_logs = len(df[df['chain_label'] == label])
            print(f"Attack Chain {label}: {chain_files} files, {total_logs} logs")
        else:
            print(f"Attack Chain {label}: 0 files, 0 logs")
    
    # Feature processing - use larger parameters
    print("\nPerforming feature processing...")
    X_seq, X_static, y_chain, tokenizer, scaler = preprocess_data(
        df, max_sequence_length=40, vocab_size=3000
    )
    
    # Check class distribution
    print("\nOriginal class distribution:")
    unique, counts = np.unique(y_chain.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Attack Chain {i}: {count} samples ({count/len(y_chain)*100:.1f}%)")
    
    # Calculate class weights - use a more balanced strategy
    from sklearn.utils.class_weight import compute_class_weight
    class_labels = y_chain.argmax(axis=1)
    classes = np.unique(class_labels)
    
    # Use slightly milder class weights
    class_weights = compute_class_weight('balanced', classes=classes, y=class_labels)
    # Limit extreme weights to avoid overcompensation
    class_weights = np.clip(class_weights, 0.5, 3.0)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"\nAdjusted class weights: {class_weight_dict}")
    
    # Build model - use attention mechanism
    model = build_improved_lstm_model(
        vocab_size=3000, 
        max_sequence_length=40, 
        static_feature_size=X_static.shape[1], 
        use_attention=True
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Train with chain-aware splitting
    print("\nStarting training (improved version)...")
    history, X_seq_test, X_static_test, y_test = train_with_chain_aware_split(
        model, X_seq, X_static, y_chain, df, class_weight_dict, epochs=150, batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(
        {'sequence_input': X_seq_test, 'static_input': X_static_test}, 
        y_test, 
        verbose=0
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed prediction analysis
    chain_prob = model.predict({'sequence_input': X_seq_test, 'static_input': X_static_test})
    predicted_classes = np.argmax(chain_prob, axis=1)
    actual_classes = np.argmax(y_test, axis=1)
    
    # Analyze prediction confidence
    prediction_confidences = np.max(chain_prob, axis=1)
    print(f"\nPrediction confidence statistics:")
    print(f"Average confidence: {np.mean(prediction_confidences):.3f}")
    print(f"Confidence standard deviation: {np.std(prediction_confidences):.3f}")
    print(f"Minimum confidence: {np.min(prediction_confidences):.3f}")
    print(f"Maximum confidence: {np.max(prediction_confidences):.3f}")
    
    # Analyze confidence by class
    for class_idx in range(3):
        class_mask = actual_classes == class_idx
        if np.any(class_mask):
            class_confidences = prediction_confidences[class_mask]
            print(f"Attack Chain {class_idx} average confidence: {np.mean(class_confidences):.3f}")
    
    # Confusion matrix and classification report
    cm = confusion_matrix(actual_classes, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(actual_classes, predicted_classes, 
                              target_names=[f'Attack Chain {i}' for i in range(3)]))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Attack Chain {i}' for i in range(3)],
                yticklabels=[f'Attack Chain {i}' for i in range(3)])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix plot saved as 'confusion_matrix.png'")
    plt.show()
    
    # Calculate and plot ROC curves
    print("\nGenerating ROC curves...")
    
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(actual_classes, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], chain_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate macro-average ROC curve and AUC
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Average and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Set colors
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    
    # Plot ROC curve for each class
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Attack Chain {i} (AUC = {roc_auc[i]:.3f})')
    
    # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"],
             color='navy', linestyle='--', linewidth=2,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # Plot random classifier baseline
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("ROC curves plot saved as 'roc_curves.png'")
    plt.show()
    
    # Print AUC statistics
    print(f"\nAUC statistics:")
    for i in range(n_classes):
        print(f"Attack Chain {i} AUC: {roc_auc[i]:.4f}")
    print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
    
    # Display prediction samples
    print(f"\nPrediction results for the first {min(15, len(y_test))} test samples:")
    for i in range(min(15, len(y_test))):
        predicted_class = predicted_classes[i]
        actual_class = actual_classes[i]
        confidence = np.max(chain_prob[i])
        is_correct = "✓" if predicted_class == actual_class else "✗"
        print(f"Sample {i+1}: Predicted={predicted_class}, Actual={actual_class}, "
              f"Confidence={confidence:.3f} {is_correct}")
    
    # Save model
    model.save('improved_chain_aware_lstm_model.keras')
    print("\nModel saved as 'improved_chain_aware_lstm_model.keras'")
    
    # Save preprocessors
    import pickle
    with open('improved_preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'tokenizer': tokenizer,
            'scaler': scaler
        }, f)
    print("Preprocessors saved as 'improved_preprocessors.pkl'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', markersize=3)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', markersize=3)
    plt.title('Model Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o', markersize=3)
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training history plot saved as 'training_history.png'")

    print("\n" + "="*50)
    print("All image files generated:")
    print("1. confusion_matrix.png - Confusion matrix heatmap")
    print("2. roc_curves.png - Multi-class ROC curves")
    print("3. training_history.png - Training history plot")
    print("="*50)
