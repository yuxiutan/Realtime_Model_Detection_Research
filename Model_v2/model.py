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

# 檢查 full_log 重複率
def check_log_duplicates(df):
    log_counts = df['full_log'].value_counts()
    total_logs = len(df)
    duplicates = log_counts[log_counts > 1]
    duplicate_rate = len(duplicates) / total_logs if total_logs > 0 else 0
    print(f"Total logs: {total_logs}")
    print(f"Unique logs: {len(log_counts)}")
    print(f"Duplicate log rate: {duplicate_rate:.2%}")
    print(f"Top 5 most frequent logs:\n{log_counts.head()}")

# 修正的資料載入函數
def load_jsonl_files_with_chain_id(file_paths):
    """載入資料並為每個攻擊鏈分配唯一ID"""
    data = []
    for i, file_path in enumerate(file_paths):
        print(f"載入 {file_path}...")
        if not os.path.exists(file_path):
            print(f"  警告: 文件 {file_path} 不存在，跳過...")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                try:
                    record = json.loads(line.strip())
                    record['chain_label'] = i  # 0, 1, 2 對應 attack_chain_0, 1, 2
                    record['is_attack'] = 1  # 所有樣本都是攻擊
                    # 每個文件就是一個完整攻擊鏈，所以所有日誌都有相同的chain_instance_id
                    record['chain_instance_id'] = f"chain_type_{i}_file_{os.path.basename(file_path)}"
                    data.append(record)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"  JSON 解析錯誤在第 {line_count + 1} 行: {e}")
                    continue
            
            print(f"  載入 {line_count} 條日誌")
    
    if not data:
        raise ValueError("沒有成功載入任何數據，請檢查文件路徑和格式")
    
    return pd.DataFrame(data)

# 改進的特徵處理函數 - 增加更多特徵工程
def preprocess_data(df, max_sequence_length=40, vocab_size=3000):  # 增加序列長度和詞彙量
    # 檢查必要欄位
    required_columns = ['@timestamp', 'agent.ip', 'agent.name', 'agent.id', 'rule.id', 'rule.mitre.id', 'full_log']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing columns in data: {missing}")
    
    # 檢查 full_log 重複率
    check_log_duplicates(df)
    
    # 處理 @timestamp - 避免 NaN 和除零問題
    df['@timestamp'] = pd.to_numeric(df['@timestamp'], errors='coerce')
    df['@timestamp'] = df['@timestamp'].fillna(df['@timestamp'].median())
    
    # 按攻擊鏈和時間排序
    df = df.sort_values(['chain_instance_id', '@timestamp']).reset_index(drop=True)
    
    # 計算時間特徵
    df['hour'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.dayofweek
    
    # 時間戳標準化
    timestamp_std = df['@timestamp'].std()
    if timestamp_std == 0 or np.isnan(timestamp_std):
        df['@timestamp_normalized'] = df['@timestamp']
    else:
        df['@timestamp_normalized'] = (df['@timestamp'] - df['@timestamp'].mean()) / timestamp_std
    
    # 計算攻擊鏈內的時間差
    df['time_diff'] = df.groupby('chain_instance_id')['@timestamp'].diff().fillna(0)
    
    # 時間差標準化
    time_diff_std = df['time_diff'].std()
    if time_diff_std == 0 or np.isnan(time_diff_std):
        df['time_diff_normalized'] = df['time_diff']
    else:
        df['time_diff_normalized'] = (df['time_diff'] - df['time_diff'].mean()) / time_diff_std
    
    # 轉換 IP 地址為整數並標準化
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
    
    # 計算日誌長度特徵
    df['log_length'] = df['full_log'].astype(str).str.len()
    df['log_length_normalized'] = (df['log_length'] - df['log_length'].mean()) / df['log_length'].std()
    
    # 計算每個攻擊鏈中的位置特徵
    df['position_in_chain'] = df.groupby('chain_instance_id').cumcount()
    df['chain_total_length'] = df.groupby('chain_instance_id')['chain_instance_id'].transform('count')
    df['position_ratio'] = df['position_in_chain'] / df['chain_total_length']
    
    # 對字串特徵進行 Label Encoding
    le_name = LabelEncoder()
    le_id = LabelEncoder()
    le_rule = LabelEncoder()
    le_mitre = LabelEncoder()
    
    df['agent.name_encoded'] = le_name.fit_transform(df['agent.name'].astype(str))
    df['agent.id_encoded'] = le_id.fit_transform(df['agent.id'].astype(str))
    df['rule.id_encoded'] = le_rule.fit_transform(df['rule.id'].astype(str))
    df['rule.mitre.id_encoded'] = le_mitre.fit_transform(df['rule.mitre.id'].astype(str))
    
    # 對 full_log 進行分詞和序列化 - 使用更大的詞彙量
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", char_level=False)
    tokenizer.fit_on_texts(df['full_log'].astype(str))
    sequences = tokenizer.texts_to_sequences(df['full_log'].astype(str))
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # 擴展靜態特徵（包含新的時間和位置特徵）
    static_features = df[['@timestamp_normalized', 'time_diff_normalized', 'agent.ip_normalized', 
                         'agent.name_encoded', 'agent.id_encoded', 'rule.id_encoded', 'rule.mitre.id_encoded',
                         'hour', 'day_of_week', 'log_length_normalized', 'position_ratio']].values
    
    # 檢查是否有 NaN 或 inf
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 使用 StandardScaler 進一步標準化
    scaler = StandardScaler()
    static_features = scaler.fit_transform(static_features)
    
    # 攻擊鏈標籤（one-hot 編碼）
    chain_labels = df['chain_label'].values
    chain_labels_onehot = np.zeros((len(chain_labels), 3))
    for i, label in enumerate(chain_labels):
        chain_labels_onehot[i, label] = 1
    
    return padded_sequences, static_features, chain_labels_onehot, tokenizer, scaler

# 改進的 LSTM 模型 - 增加複雜度和注意力機制
def build_improved_lstm_model(vocab_size, max_sequence_length, static_feature_size, use_attention=True):
    # 序列輸入 (full_log)
    sequence_input = tf.keras.Input(shape=(max_sequence_length,), name='sequence_input')
    
    # 更大的 Embedding 維度
    embedding = tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True)(sequence_input)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    
    # 多層 Bidirectional LSTM
    lstm1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(embedding)
    
    lstm2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True if use_attention else False, 
                           dropout=0.3, recurrent_dropout=0.2)
    )(lstm1)
    
    if use_attention:
        # 添加注意力機制
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(lstm2, lstm2)
        attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
        lstm_output = attention
    else:
        lstm_output = lstm2
    
    # 靜態特徵輸入 - 更深的網絡
    static_input = tf.keras.Input(shape=(static_feature_size,), name='static_input')
    static_dense1 = tf.keras.layers.Dense(64, activation='relu')(static_input)
    static_dense1 = tf.keras.layers.BatchNormalization()(static_dense1)
    static_dense1 = tf.keras.layers.Dropout(0.3)(static_dense1)
    
    static_dense2 = tf.keras.layers.Dense(32, activation='relu')(static_dense1)
    static_dense2 = tf.keras.layers.BatchNormalization()(static_dense2)
    static_dense2 = tf.keras.layers.Dropout(0.2)(static_dense2)
    
    # 合併 LSTM 和靜態特徵
    combined = tf.keras.layers.Concatenate()([lstm_output, static_dense2])
    
    # 更深的分類網絡
    dense1 = tf.keras.layers.Dense(256, activation='relu')(combined)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.4)(dense1)
    
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    dense3 = tf.keras.layers.Dense(64, activation='relu')(dense2)
    dense3 = tf.keras.layers.BatchNormalization()(dense3)
    dense3 = tf.keras.layers.Dropout(0.2)(dense3)
    
    # 輸出層
    chain_output = tf.keras.layers.Dense(3, activation='softmax', name='chain_output')(dense3)
    
    model = tf.keras.Model(inputs=[sequence_input, static_input], outputs=chain_output)
    
    # 使用更好的優化器設置
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,  # 稍微高一點的初始學習率
        weight_decay=0.01,    # 權重衰減
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def chain_aware_train_test_split(df, test_size=0.2, random_state=42):
    """基於攻擊鏈文件進行分割，確保同一文件的所有日誌在同一集合中"""
    
    # 獲取所有唯一的攻擊鏈文件
    unique_chains = df['chain_instance_id'].unique()
    
    print(f"總共有 {len(unique_chains)} 個攻擊鏈文件:")
    for chain_id in unique_chains:
        chain_data = df[df['chain_instance_id'] == chain_id]
        chain_label = chain_data['chain_label'].iloc[0]
        log_count = len(chain_data)
        print(f"  {chain_id}: Attack Chain {chain_label}, {log_count} 條日誌")
    
    # 按攻擊鏈標籤分組文件
    chain_groups = {}
    for chain_id in unique_chains:
        chain_data = df[df['chain_instance_id'] == chain_id]
        chain_label = chain_data['chain_label'].iloc[0]
        
        if chain_label not in chain_groups:
            chain_groups[chain_label] = []
        chain_groups[chain_label].append(chain_id)
    
    print(f"\n攻擊鏈類型分布:")
    for label, chains in chain_groups.items():
        print(f"  Attack Chain {label}: {len(chains)} 個文件")
    
    # 分層分割策略 - 確保測試集有足夠的樣本
    train_indices = []
    test_indices = []
    
    np.random.seed(random_state)
    
    for label, chain_files in chain_groups.items():
        if len(chain_files) == 1:
            # 如果該類型只有一個文件，採用更保守的分割比例
            chain_id = chain_files[0]
            chain_indices = df[df['chain_instance_id'] == chain_id].index.tolist()
            
            # 隨機打亂索引
            np.random.shuffle(chain_indices)
            
            # 確保測試集至少有5個樣本或20%，取較大者
            n_test = max(5, int(len(chain_indices) * test_size))
            n_test = min(n_test, len(chain_indices) - 10)  # 但不能太多，要留足夠的訓練數據
            n_train = len(chain_indices) - n_test
            
            test_indices.extend(chain_indices[:n_test])
            train_indices.extend(chain_indices[n_test:])
            
            print(f"    分割 {chain_id}: {n_train} 訓練日誌, {n_test} 測試日誌")
        else:
            # 如果有多個文件，就按文件分割
            chains = np.array(chain_files)
            np.random.shuffle(chains)
            
            n_test_files = max(1, int(len(chains) * test_size))
            n_train_files = len(chains) - n_test_files
            
            test_files = chains[:n_test_files]
            train_files = chains[n_test_files:]
            
            # 收集對應的索引
            for chain_id in test_files:
                test_indices.extend(df[df['chain_instance_id'] == chain_id].index.tolist())
            
            for chain_id in train_files:
                train_indices.extend(df[df['chain_instance_id'] == chain_id].index.tolist())
            
            print(f"    Attack Chain {label}: {n_train_files} 訓練文件, {n_test_files} 測試文件")
    
    # 創建訓練和測試集
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    print(f"\n最終分割結果:")
    print(f"訓練集: {len(train_df)} 條日誌")
    print(f"測試集: {len(test_df)} 條日誌")
    
    # 檢查分割後的類別分布
    print("\n訓練集類別分布:")
    train_counts = train_df['chain_label'].value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Attack Chain {label}: {count} 條日誌")
    
    print("測試集類別分布:")
    test_counts = test_df['chain_label'].value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Attack Chain {label}: {count} 條日誌")
    
    return train_df, test_df

def train_with_chain_aware_split(model, X_seq, X_static, y_chain, df, class_weight_dict, epochs=150, batch_size=32):
    """使用攻擊鏈感知分割進行訓練"""
    
    # 進行攻擊鏈感知的分割
    train_df, test_df = chain_aware_train_test_split(df, test_size=0.2, random_state=42)
    
    # 獲取對應的索引
    train_indices = train_df.index.tolist()
    test_indices = test_df.index.tolist()
    
    # 分割特徵和標籤
    X_seq_train = X_seq[train_indices]
    X_static_train = X_static[train_indices]
    y_train = y_chain[train_indices]
    
    X_seq_test = X_seq[test_indices]
    X_static_test = X_static[test_indices]
    y_test = y_chain[test_indices]
    
    # 從訓練集中再分出驗證集
    val_split = 0.15  # 減少驗證集比例，增加訓練數據
    n_val = int(len(X_seq_train) * val_split)
    
    # 隨機選擇驗證集索引
    np.random.seed(42)
    val_indices = np.random.choice(len(X_seq_train), n_val, replace=False)
    train_indices_inner = np.setdiff1d(range(len(X_seq_train)), val_indices)
    
    X_seq_train_final = X_seq_train[train_indices_inner]
    X_static_train_final = X_static_train[train_indices_inner]
    y_train_final = y_train[train_indices_inner]
    
    X_seq_val = X_seq_train[val_indices]
    X_static_val = X_static_train[val_indices]
    y_val = y_train[val_indices]
    
    # 打印最終的數據分布
    print("\n最終數據分布:")
    print("訓練集類別分布:")
    unique, counts = np.unique(y_train_final.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  Attack Chain {i}: {count} samples")
        
    print("驗證集類別分布:")
    unique, counts = np.unique(y_val.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  Attack Chain {i}: {count} samples")
        
    print("測試集類別分布:")
    unique, counts = np.unique(y_test.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  Attack Chain {i}: {count} samples")
    
    # 改進的回調函數
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # 增加耐心值
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # 添加最小改進閾值
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3,  # 更激進的學習率縮減
            patience=10, 
            min_lr=1e-8,
            verbose=1
        ),
        # 添加學習率調度器
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.95 ** epoch), verbose=0
        )
    ]
    
    # 訓練模型
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

# 主程式
if __name__ == "__main__":
    # 設置隨機種子以確保可重現性
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 讀取數據
    file_paths = ['attack_chain_0.jsonl', 'attack_chain_1.jsonl', 'attack_chain_2.jsonl']
    df = load_jsonl_files_with_chain_id(file_paths)
    
    # 檢查攻擊鏈文件統計
    print("攻擊鏈文件統計:")
    for label in [0, 1, 2]:
        if label in df['chain_label'].values:
            chain_files = df[df['chain_label'] == label]['chain_instance_id'].nunique()
            total_logs = len(df[df['chain_label'] == label])
            print(f"Attack Chain {label}: {chain_files} 個文件, {total_logs} 條日誌")
        else:
            print(f"Attack Chain {label}: 0 個文件, 0 條日誌")
    
    # 特徵處理 - 使用更大的參數
    print("\n進行特徵處理...")
    X_seq, X_static, y_chain, tokenizer, scaler = preprocess_data(
        df, max_sequence_length=40, vocab_size=3000
    )
    
    # 檢查類別分布
    print("\n原始類別分布:")
    unique, counts = np.unique(y_chain.argmax(axis=1), return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Attack Chain {i}: {count} samples ({count/len(y_chain)*100:.1f}%)")
    
    # 計算類別權重 - 使用更平衡的策略
    from sklearn.utils.class_weight import compute_class_weight
    class_labels = y_chain.argmax(axis=1)
    classes = np.unique(class_labels)
    
    # 使用稍微溫和的類別權重
    class_weights = compute_class_weight('balanced', classes=classes, y=class_labels)
    # 限制權重的極值，避免過度補償
    class_weights = np.clip(class_weights, 0.5, 3.0)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"\n調整後的類別權重: {class_weight_dict}")
    
    # 建立模型 - 使用注意力機制
    model = build_improved_lstm_model(
        vocab_size=3000, 
        max_sequence_length=40, 
        static_feature_size=X_static.shape[1], 
        use_attention=True
    )
    
    print("\n模型架構:")
    model.summary()
    
    # 使用攻擊鏈感知分割進行訓練
    print("\n開始訓練（改進版本）...")
    history, X_seq_test, X_static_test, y_test = train_with_chain_aware_split(
        model, X_seq, X_static, y_chain, df, class_weight_dict, epochs=150, batch_size=32
    )
    
    # 評估模型
    print("\n評估模型...")
    test_loss, test_accuracy = model.evaluate(
        {'sequence_input': X_seq_test, 'static_input': X_static_test}, 
        y_test, 
        verbose=0
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 詳細的預測分析
    chain_prob = model.predict({'sequence_input': X_seq_test, 'static_input': X_static_test})
    predicted_classes = np.argmax(chain_prob, axis=1)
    actual_classes = np.argmax(y_test, axis=1)
    
    # 分析預測信心值
    prediction_confidences = np.max(chain_prob, axis=1)
    print(f"\n預測信心值統計:")
    print(f"平均信心值: {np.mean(prediction_confidences):.3f}")
    print(f"信心值標準差: {np.std(prediction_confidences):.3f}")
    print(f"最低信心值: {np.min(prediction_confidences):.3f}")
    print(f"最高信心值: {np.max(prediction_confidences):.3f}")
    
    # 按類別分析信心值
    for class_idx in range(3):
        class_mask = actual_classes == class_idx
        if np.any(class_mask):
            class_confidences = prediction_confidences[class_mask]
            print(f"Attack Chain {class_idx} 平均信心值: {np.mean(class_confidences):.3f}")
    
    # 混淆矩陣和分類報告
    cm = confusion_matrix(actual_classes, predicted_classes)
    print("\n混淆矩陣:")
    print(cm)
    
    print("\n分類報告:")
    print(classification_report(actual_classes, predicted_classes, 
                              target_names=[f'Attack Chain {i}' for i in range(3)]))
    
    # 繪製混淆矩陣圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Attack Chain {i}' for i in range(3)],
                yticklabels=[f'Attack Chain {i}' for i in range(3)])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩陣圖片已保存為 'confusion_matrix.png'")
    plt.show()
    
    # 計算並繪製ROC曲線
    print("\n生成ROC曲線...")
    
    # 將標籤進行二值化處理（用於多類別ROC）
    y_test_bin = label_binarize(actual_classes, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # 計算每個類別的ROC曲線和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], chain_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 計算macro-average ROC曲線和AUC
    # 首先聚合所有假正率
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # 然後在這些點上插值所有ROC曲線
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # 最後求平均並計算AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 繪製ROC曲線
    plt.figure(figsize=(10, 8))
    
    # 設定顏色
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    
    # 繪製每個類別的ROC曲線
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Attack Chain {i} (AUC = {roc_auc[i]:.3f})')
    
    # 繪製macro-average ROC曲線
    plt.plot(fpr["macro"], tpr["macro"],
             color='navy', linestyle='--', linewidth=2,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # 繪製隨機分類器的基準線
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
    print("ROC曲線圖片已保存為 'roc_curves.png'")
    plt.show()
    
    # 打印AUC值統計
    print(f"\nAUC值統計:")
    for i in range(n_classes):
        print(f"Attack Chain {i} AUC: {roc_auc[i]:.4f}")
    print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
    
    # 顯示預測樣本
    print(f"\n前{min(15, len(y_test))}個測試樣本的預測結果:")
    for i in range(min(15, len(y_test))):
        predicted_class = predicted_classes[i]
        actual_class = actual_classes[i]
        confidence = np.max(chain_prob[i])
        is_correct = "✓" if predicted_class == actual_class else "✗"
        print(f"Sample {i+1}: Predicted={predicted_class}, Actual={actual_class}, "
              f"Confidence={confidence:.3f} {is_correct}")
    
    # 保存模型
    model.save('improved_chain_aware_lstm_model.keras')
    print("\n模型已保存為 'improved_chain_aware_lstm_model.keras'")
    
    # 保存預處理器
    import pickle
    with open('improved_preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'tokenizer': tokenizer,
            'scaler': scaler
        }, f)
    print("預處理器已保存為 'improved_preprocessors.pkl'")
    
    # 繪製訓練歷史
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
    
    print("訓練歷史圖表已保存為 'training_history.png'")

    print("\n" + "="*50)
    print("所有圖片文件已生成:")
    print("1. confusion_matrix.png - 混淆矩陣熱力圖")
    print("2. roc_curves.png - 多類別ROC曲線圖")
    print("3. training_history.png - 訓練歷史圖表")
    print("="*50)
