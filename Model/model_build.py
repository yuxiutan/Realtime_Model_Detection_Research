import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ipaddress
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import random

# Check full_log duplicate rate
def check_log_duplicates(df):
    log_counts = df['full_log'].value_counts()
    total_logs = len(df)
    duplicates = log_counts[log_counts > 1]
    duplicate_rate = len(duplicates) / total_logs if total_logs > 0 else 0
    print(f"Total logs: {total_logs}")
    print(f"Unique logs: {len(log_counts)}")
    print(f"Duplicate log rate: {duplicate_rate:.2%}")
    print(f"Top 5 most frequent logs:\n{log_counts.head()}")

# Text augmentation function
def augment_log(log):
    words = str(log).split()
    if random.random() < 0.3:
        words.insert(random.randint(0, len(words)), "INFO")
    if random.random() < 0.2 and len(words) > 1:
        words = words[::-1]  # Simple word order reversal
    return " ".join(words)

# Improved data loading function
def load_jsonl_files_with_chain_id(file_paths):
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
                    record['chain_label'] = i
                    record['is_attack'] = 1
                    record['chain_instance_id'] = f"chain_type_{i}_file_{os.path.basename(file_path)}"
                    data.append(record)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"  JSON parsing error at line {line_count + 1}: {e}")
                    continue
            print(f"  Loaded {line_count} logs")
    
    if not data:
        raise ValueError("No data loaded successfully, please check file paths and format")
    
    df = pd.DataFrame(data)
    
    # Simple oversampling and text augmentation for Attack Chain 1
    attack_chain_1 = df[df['chain_label'] == 1]
    if not attack_chain_1.empty:
        oversampled_chain_1 = attack_chain_1.sample(n=200, replace=True, random_state=42)
        oversampled_chain_1['full_log'] = oversampled_chain_1['full_log'].apply(augment_log)
        df = pd.concat([df, oversampled_chain_1], ignore_index=True)
    
    return df

# Improved feature processing function
def preprocess_data(df, max_sequence_length=60, vocab_size=5000):
    required_columns = ['@timestamp', 'agent.ip', 'agent.name', 'agent.id', 'rule.id', 'rule.mitre.id', 'full_log']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing columns in data: {missing}")
    
    check_log_duplicates(df)
    
    # Handle @timestamp
    df['@timestamp'] = pd.to_numeric(df['@timestamp'], errors='coerce')
    df['@timestamp'] = df['@timestamp'].fillna(df['@timestamp'].median())
    
    # Sort by attack chain and timestamp
    df = df.sort_values(['chain_instance_id', '@timestamp']).reset_index(drop=True)
    
    # Time features
    df['hour'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.dayofweek
    timestamp_std = df['@timestamp'].std()
    if timestamp_std == 0 or np.isnan(timestamp_std):
        df['@timestamp_normalized'] = df['@timestamp']
    else:
        df['@timestamp_normalized'] = (df['@timestamp'] - df['@timestamp'].mean()) / timestamp_std
    
    # Time difference
    df['time_diff'] = df.groupby('chain_instance_id')['@timestamp'].diff().fillna(0)
    time_diff_std = df['time_diff'].std()
    if time_diff_std == 0 or np.isnan(time_diff_std):
        df['time_diff_normalized'] = df['time_diff']
    else:
        df['time_diff_normalized'] = (df['time_diff'] - df['time_diff'].mean()) / time_diff_std
    
    # Convert IP address to integer
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
    
    # Log length
    df['log_length'] = df['full_log'].astype(str).str.len()
    log_length_std = df['log_length'].std()
    if log_length_std == 0 or np.isnan(log_length_std):
        df['log_length_normalized'] = df['log_length']
    else:
        df['log_length_normalized'] = (df['log_length'] - df['log_length'].mean()) / log_length_std
    
    # Attack chain position
    df['position_in_chain'] = df.groupby('chain_instance_id').cumcount()
    df['chain_total_length'] = df.groupby('chain_instance_id')['chain_instance_id'].transform('count')
    df['position_ratio'] = df['position_in_chain'] / df['chain_total_length']
    
    # Contextual features
    # Ensure rule.id is consistently string
    df['rule.id'] = df['rule.id'].astype(str)
    df['prev_rule_id'] = df.groupby('chain_instance_id')['rule.id'].shift(1).fillna('NONE')
    df['next_rule_id'] = df.groupby('chain_instance_id')['rule.id'].shift(-1).fillna('NONE')
    
    # MITRE ATT&CK features
    df['mitre_tactic'] = df['rule.mitre.id'].apply(lambda x: str(x).split('.')[0] if isinstance(x, (str, int, float)) else 'UNKNOWN')
    df['mitre_technique'] = df['rule.mitre.id'].apply(lambda x: str(x) if isinstance(x, (str, int, float)) else 'UNKNOWN')
    
    # Ensure all columns to be encoded are strings
    df['agent.name'] = df['agent.name'].astype(str)
    df['agent.id'] = df['agent.id'].astype(str)
    
    # Label Encoding
    le_name = LabelEncoder()
    le_id = LabelEncoder()
    le_rule = LabelEncoder()
    le_mitre_tactic = LabelEncoder()
    le_mitre_technique = LabelEncoder()
    le_prev_rule = LabelEncoder()
    le_next_rule = LabelEncoder()
    
    df['agent.name_encoded'] = le_name.fit_transform(df['agent.name'])
    df['agent.id_encoded'] = le_id.fit_transform(df['agent.id'])
    df['rule.id_encoded'] = le_rule.fit_transform(df['rule.id'])
    df['mitre_tactic_encoded'] = le_mitre_tactic.fit_transform(df['mitre_tactic'])
    df['mitre_technique_encoded'] = le_mitre_technique.fit_transform(df['mitre_technique'])
    df['prev_rule_id_encoded'] = le_prev_rule.fit_transform(df['prev_rule_id'])
    df['next_rule_id_encoded'] = le_next_rule.fit_transform(df['next_rule_id'])
    
    # N-gram features
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=100)
    ngram_features = vectorizer.fit_transform(df['full_log'].astype(str)).toarray()
    ngram_columns = vectorizer.get_feature_names_out()
    df_ngram = pd.DataFrame(ngram_features, columns=[f'ngram_{col}' for col in ngram_columns])
    df = pd.concat([df, df_ngram], axis=1)
    
    # Character-level tokenization
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", char_level=True)
    tokenizer.fit_on_texts(df['full_log'].astype(str))
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
    
    scaler = StandardScaler()
    static_features = scaler.fit_transform(static_features)
    
    # Attack chain labels
    chain_labels = df['chain_label'].values
    chain_labels_onehot = np.zeros((len(chain_labels), 3))
    for i, label in enumerate(chain_labels):
        chain_labels_onehot[i, label] = 1
    
    return padded_sequences, static_features, chain_labels_onehot, tokenizer, scaler, df, vectorizer

# Focal Loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * y_true * tf.pow(1. - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fn

# Transformer model
def build_transformer_model(vocab_size, max_sequence_length, static_feature_size):
    sequence_input = tf.keras.Input(shape=(max_sequence_length,), name='sequence_input')
    embedding = tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True)(sequence_input)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    
    # Transformer encoder layer
    transformer_block = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(embedding, embedding)
    transformer_block = tf.keras.layers.LayerNormalization(epsilon=1e-6)(transformer_block + embedding)
    transformer_block = tf.keras.layers.Dense(128, activation='relu', 
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(transformer_block)
    transformer_block = tf.keras.layers.LayerNormalization(epsilon=1e-6)(transformer_block)
    transformer_output = tf.keras.layers.GlobalAveragePooling1D()(transformer_block)
    
    # Static features
    static_input = tf.keras.Input(shape=(static_feature_size,), name='static_input')
    static_dense = tf.keras.layers.Dense(64, activation='relu', 
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(static_input)
    static_dense = tf.keras.layers.BatchNormalization()(static_dense)
    static_dense = tf.keras.layers.Dropout(0.2)(static_dense)
    
    # Combine
    combined = tf.keras.layers.Concatenate()([transformer_output, static_dense])
    dense1 = tf.keras.layers.Dense(128, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(3, activation='softmax', name='chain_output')(dense1)
    
    model = tf.keras.Model(inputs=[sequence_input, static_input], outputs=output)
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
    return model

# Chain-aware train-test split
def chain_aware_train_test_split(df, test_size=0.3, random_state=42):
    unique_chains = df['chain_instance_id'].unique()
    print(f"Total {len(unique_chains)} attack chain files:")
    for chain_id in unique_chains:
        chain_data = df[df['chain_instance_id'] == chain_id]
        chain_label = chain_data['chain_label'].iloc[0]
        log_count = len(chain_data)
        print(f"  {chain_id}: Attack Chain {chain_label}, {log_count} logs")
    
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
    
    train_indices = []
    test_indices = []
    np.random.seed(random_state)
    
    for label, chain_files in chain_groups.items():
        if len(chain_files) == 1:
            chain_id = chain_files[0]
            chain_indices = df[df['chain_instance_id'] == chain_id].index.tolist()
            np.random.shuffle(chain_indices)
            n_test = max(20 if label == 1 else 5, int(len(chain_indices) * test_size))
            n_test = min(n_test, len(chain_indices) - 10)
            n_train = len(chain_indices) - n_test
            test_indices.extend(chain_indices[:n_test])
            train_indices.extend(chain_indices[n_test:])
            print(f"    Split {chain_id}: {n_train} training logs, {n_test} test logs")
        else:
            chains = np.array(chain_files)
            np.random.shuffle(chains)
            n_test_files = max(1, int(len(chains) * test_size))
            n_train_files = len(chains) - n_test_files
            test_files = chains[:n_test_files]
            train_files = chains[n_test_files:]
            for chain_id in test_files:
                test_indices.extend(df[df['chain_instance_id'] == chain_id].index.tolist())
            for chain_id in train_files:
                train_indices.extend(df[df['chain_instance_id'] == chain_id].index.tolist())
            print(f"    Attack Chain {label}: {n_train_files} training files, {n_test_files} test files")
    
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    print(f"\nFinal split results:")
    print(f"Training set: {len(train_df)} logs")
    print(f"Test set: {len(test_df)} logs")
    
    print("\nTraining set class distribution:")
    train_counts = train_df['chain_label'].value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Attack Chain {label}: {count} logs")
    
    print("Test set class distribution:")
    test_counts = test_df['chain_label'].value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Attack Chain {label}: {count} logs")
    
    return train_df, test_df

# Cross-validation training
def train_with_cross_validation(model, X_seq, X_static, y_chain, df, class_weight_dict, epochs=100, batch_size=32):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    histories = []
    
    train_df, test_df = chain_aware_train_test_split(df, test_size=0.3, random_state=42)
    train_indices = train_df.index.tolist()
    test_indices = test_df.index.tolist()
    
    X_seq_test = X_seq[test_indices]
    X_static_test = X_static[test_indices]
    y_test = y_chain[test_indices]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['chain_label'])):
        print(f"\nTraining fold {fold + 1}...")
        # Convert relative indices to absolute indices
        train_idx_abs = [train_indices[i] for i in train_idx]
        val_idx_abs = [train_indices[i] for i in val_idx]
        
        X_seq_train = X_seq[train_idx_abs]
        X_static_train = X_static[train_idx_abs]
        y_train = y_chain[train_idx_abs]
        X_seq_val = X_seq[val_idx_abs]
        X_static_val = X_static[val_idx_abs]
        y_val = y_chain[val_idx_abs]
        
        # Print data distribution
        print("Training set class distribution:")
        unique, counts = np.unique(y_train.argmax(axis=1), return_counts=True)
        for i, count in zip(unique, counts):
            print(f"  Attack Chain {i}: {count} samples")
        print("Validation set class distribution:")
        unique, counts = np.unique(y_val.argmax(axis=1), return_counts=True)
        for i, count in zip(unique, counts):
            print(f"  Attack Chain {i}: {count} samples")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1, min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8, verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            {'sequence_input': X_seq_train, 'static_input': X_static_train},
            y_train,
            validation_data=({'sequence_input': X_seq_val, 'static_input': X_static_val}, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        histories.append(history)
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(
            {'sequence_input': X_seq_test, 'static_input': X_static_test}, y_test, verbose=0
        )
        print(f"Fold {fold+1} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return histories, X_seq_test, X_static_test, y_test, test_df, train_df

# Main program
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    file_paths = ['attack_chain_0.jsonl', 'attack_chain_1.jsonl', 'attack_chain_2.jsonl']
    df = load_jsonl_files_with_chain_id(file_paths)
    
    # Check data distribution
    print("Attack chain file statistics:")
    for label in [0, 1, 2]:
        if label in df['chain_label'].values:
            chain_files = df[df['chain_label'] == label]['chain_instance_id'].nunique()
            total_logs = len(df[df['chain_label'] == label])
            print(f"Attack Chain {label}: {chain_files} files, {total_logs} logs")
        else:
            print(f"Attack Chain {label}: 0 files, 0 logs")
    
    # Check Attack Chain 1 logs
    print("\nAttack Chain 1 log contents:")
    print(df[df['chain_label'] == 1]['full_log'].value_counts())
    
    # Check rule.id data types
    print("\nrule.id data type distribution:")
    print(df['rule.id'].apply(type).value_counts())
    
    # Feature processing
    print("\nPerforming feature processing...")
    X_seq, X_static, y_chain, tokenizer, scaler, df, vectorizer = preprocess_data(
        df, max_sequence_length=60, vocab_size=5000
    )
    
    # Class distribution
    print("\nOriginal class distribution:")
    unique, counts = np.unique(y_chain.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Attack Chain {i}: {count} samples ({count/len(y_chain)*100:.1f}%)")
    
    # Class weights
    class_labels = y_chain.argmax(axis=1)
    classes = np.unique(class_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=class_labels)
    class_weight_dict = {0: 0.7, 1: 5.0, 2: 0.7}  # Manually adjust weights for Attack Chain 1
    print(f"\nAdjusted class weights: {class_weight_dict}")
    
    # Build model
    model = build_transformer_model(
        vocab_size=5000,
        max_sequence_length=60,
        static_feature_size=X_static.shape[1]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Training
    print("\nStarting cross-validation training...")
    histories, X_seq_test, X_static_test, y_test, test_df, train_df = train_with_cross_validation(
        model, X_seq, X_static, y_chain, df, class_weight_dict, epochs=100, batch_size=32
    )
    
    # Evaluation
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(
        {'sequence_input': X_seq_test, 'static_input': X_static_test}, y_test, verbose=0
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # XGBoost ensemble
    print("\nTraining XGBoost model...")
    xgb_model = XGBClassifier(random_state=42)
    train_indices = train_df.index.tolist()
    xgb_model.fit(X_static[train_indices], y_chain[train_indices].argmax(axis=1))
    xgb_model.save_model('xgb_model.json')
    print("XGBoost model saved as 'xgb_model.json'")
    xgb_pred = xgb_model.predict_proba(X_static_test)
    
    # Transformer predictions
    transformer_pred = model.predict({'sequence_input': X_seq_test, 'static_input': X_static_test})
    
    # Voting ensemble
    ensemble_pred = (transformer_pred + xgb_pred) / 2
    predicted_classes = np.argmax(ensemble_pred, axis=1)
    actual_classes = np.argmax(y_test, axis=1)
    
    # Prediction confidence
    prediction_confidences = np.max(ensemble_pred, axis=1)
    print(f"\nPrediction confidence statistics:")
    print(f"Average confidence: {np.mean(prediction_confidences):.3f}")
    print(f"Confidence standard deviation: {np.std(prediction_confidences):.3f}")
    print(f"Minimum confidence: {np.min(prediction_confidences):.3f}")
    print(f"Maximum confidence: {np.max(prediction_confidences):.3f}")
    
    for class_idx in range(3):
        class_mask = actual_classes == class_idx
        if np.any(class_mask):
            class_confidences = prediction_confidences[class_mask]
            print(f"Attack Chain {class_idx} average confidence: {np.mean(class_confidences):.3f}")
    
    # Error analysis
    test_df['predicted_class'] = predicted_classes
    test_df['actual_class'] = actual_classes
    errors = test_df[(test_df['chain_label'] == 1) & (test_df['predicted_class'] != test_df['actual_class'])]
    print("\nAttack Chain 1 misclassified samples:")
    print(errors[['full_log', 'agent.ip', 'rule.id', 'rule.mitre.id']])
    
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
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    # ROC curves
    print("\nGenerating ROC curves...")
    y_test_bin = label_binarize(actual_classes, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], ensemble_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Attack Chain {i} (AUC = {roc_auc[i]:.3f})')
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', linewidth=2,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
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
    print("ROC curves saved as 'roc_curves.png'")
    plt.close()
    
    # AUC statistics
    print(f"\nAUC statistics:")
    for i in range(n_classes):
        print(f"Attack Chain {i} AUC: {roc_auc[i]:.4f}")
    print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
    
    # Test sample predictions
    print(f"\nPredictions for first {min(15, len(y_test))} test samples:")
    for i in range(min(15, len(y_test))):
        predicted_class = predicted_classes[i]
        actual_class = actual_classes[i]
        confidence = np.max(ensemble_pred[i])
        is_correct = "✓" if predicted_class == actual_class else "✗"
        print(f"Sample {i+1}: Predicted={predicted_class}, Actual={actual_class}, "
              f"Confidence={confidence:.3f} {is_correct}")
    
    # Save model
    model.save('improved_transformer_model.keras')
    print("\nModel saved as 'improved_transformer_model.keras'")
    
    # Save preprocessors
    import pickle
    with open('improved_preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'tokenizer': tokenizer,
            'scaler': scaler,
            'vectorizer': vectorizer
        }, f)
    print("Preprocessors saved as 'improved_preprocessors.pkl'")
    
    # Training history
    plt.figure(figsize=(12, 4))
    for fold, history in enumerate(histories):
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label=f'Fold {fold+1} Training Accuracy', marker='o', markersize=3)
        plt.plot(history.history['val_accuracy'], label=f'Fold {fold+1} Validation Accuracy', marker='s', markersize=3)
        plt.title('Model Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label=f'Fold {fold+1} Training Loss', marker='o', markersize=3)
        plt.plot(history.history['val_loss'], label=f'Fold {fold+1} Validation Loss', marker='s', markersize=3)
        plt.title('Model Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.close()
    
    print("\n" + "="*50)
    print("All generated image files:")
    print("1. confusion_matrix.png - Confusion matrix heatmap")
    print("2. roc_curves.png - Multi-class ROC curves")
    print("3. training_history.png - Training history plots")
    print("="*50)
