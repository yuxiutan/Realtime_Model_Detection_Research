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

# 載入保存的模型和預處理器
def load_model_and_preprocessors(model_path='improved_chain_aware_lstm_model.keras', 
                                preprocessor_path='improved_preprocessors.pkl'):
    """載入訓練好的模型和預處理器"""
    print("載入模型和預處理器...")
    try:
        model = tf.keras.models.load_model(model_path)
        with open(preprocessor_path, 'rb') as f:
            preprocessors = pickle.load(f)
        tokenizer = preprocessors['tokenizer']
        scaler = preprocessors['scaler']
        print("模型和預處理器載入成功！")
        return model, tokenizer, scaler
    except Exception as e:
        raise ValueError(f"載入模型或預處理器失敗: {e}")

# 載入新數據
def load_new_jsonl_file(file_path, chain_instance_id="test_chain"):
    """載入新的 JSONL 文件並為其分配一個 chain_instance_id"""
    print(f"載入新數據文件 {file_path}...")
    if not os.path.exists(file_path):
        raise ValueError(f"文件 {file_path} 不存在")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                record['chain_instance_id'] = chain_instance_id  # 為新數據分配一個唯一的 chain_instance_id
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"JSON 解析錯誤: {e}")
                continue
    
    if not data:
        raise ValueError("新數據文件中沒有有效數據")
    
    df = pd.DataFrame(data)
    print(f"成功載入 {len(df)} 條日誌")
    return df

# 預處理新數據
def preprocess_new_data(df, tokenizer, scaler, max_sequence_length=40):
    """對新數據進行與訓練時一致的預處理"""
    print("開始預處理新數據...")
    
    # 檢查必要欄位
    required_columns = ['@timestamp', 'agent.ip', 'agent.name', 'agent.id', 'rule.id', 'rule.mitre.id', 'full_log']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing columns in new data: {missing}")
    
    # 按 chain_instance_id 和時間排序
    df = df.sort_values(['chain_instance_id', '@timestamp']).reset_index(drop=True)
    
    # 處理 @timestamp
    df['@timestamp'] = pd.to_numeric(df['@timestamp'], errors='coerce')
    df['@timestamp'] = df['@timestamp'].fillna(df['@timestamp'].median())
    
    # 提取時間特徵
    df['hour'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['@timestamp'], unit='ms').dt.dayofweek
    
    # 標準化時間戳
    timestamp_std = df['@timestamp'].std()
    if timestamp_std == 0 or np.isnan(timestamp_std):
        df['@timestamp_normalized'] = df['@timestamp']
    else:
        df['@timestamp_normalized'] = (df['@timestamp'] - df['@timestamp'].mean()) / timestamp_std
    
    # 計算時間差
    df['time_diff'] = df.groupby('chain_instance_id')['@timestamp'].diff().fillna(0)
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
    
    # 計算攻擊鏈中的位置特徵
    df['position_in_chain'] = df.groupby('chain_instance_id').cumcount()
    df['chain_total_length'] = df.groupby('chain_instance_id')['chain_instance_id'].transform('count')
    df['position_ratio'] = df['position_in_chain'] / df['chain_total_length']
    
    # Label Encoding（使用訓練時的編碼器，假設已保存）
    # 這裡假設新數據的類別值在訓練數據中已經見過
    # 如果有新值，可能需要處理未知類別（例如設為默認值）
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
    
    # 對 full_log 進行分詞和序列化
    sequences = tokenizer.texts_to_sequences(df['full_log'].astype(str))
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # 提取靜態特徵
    static_features = df[['@timestamp_normalized', 'time_diff_normalized', 'agent.ip_normalized', 
                         'agent.name_encoded', 'agent.id_encoded', 'rule.id_encoded', 'rule.mitre.id_encoded',
                         'hour', 'day_of_week', 'log_length_normalized', 'position_ratio']].values
    
    # 處理 NaN 和無限值
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 使用訓練時的 scaler 進行標準化
    static_features = scaler.transform(static_features)
    
    return padded_sequences, static_features, df

# 進行模型推論
def perform_inference(model, X_seq, X_static, df, output_file='inference_results.csv'):
    """對新數據進行推論並保存結果"""
    print("開始模型推論...")
    
    # 進行預測
    predictions = model.predict({'sequence_input': X_seq, 'static_input': X_static}, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    prediction_confidences = np.max(predictions, axis=1)
    
    # 整理結果
    results = pd.DataFrame({
        'chain_instance_id': df['chain_instance_id'],
        '@timestamp': df['@timestamp'],
        'full_log': df['full_log'],
        'predicted_chain': predicted_classes,
        'confidence': prediction_confidences
    })
    
    # 添加各類別的概率
    for i in range(3):
        results[f'prob_chain_{i}'] = predictions[:, i]
    
    # 打印前幾個預測結果
    print(f"\n前{min(10, len(results))}個預測結果：")
    for idx, row in results.head(10).iterrows():
        print(f"Sample {idx+1}:")
        print(f"  Chain Instance ID: {row['chain_instance_id']}")
        print(f"  Timestamp: {row['@timestamp']}")
        print(f"  Log: {row['full_log'][:100]}...")  # 只顯示前100個字符
        print(f"  Predicted Attack Chain: {row['predicted_chain']}")
        print(f"  Confidence: {row['confidence']:.3f}")
        print(f"  Probabilities: Chain 0={row['prob_chain_0']:.3f}, "
              f"Chain 1={row['prob_chain_1']:.3f}, Chain 2={row['prob_chain_2']:.3f}")
        print("-" * 50)
    
    # 按 chain_instance_id 統計預測結果
    chain_summary = results.groupby(['chain_instance_id', 'predicted_chain']).size().unstack(fill_value=0)
    print("\n每個攻擊鏈的預測分佈：")
    print(chain_summary)
    
    # 可視化信心值分佈
    plt.figure(figsize=(10, 6))
    sns.histplot(results['confidence'], bins=30, kde=True)
    plt.title('Prediction Confidence Distribution', fontsize=14)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("信心值分佈圖已保存為 'confidence_distribution.png'")
    plt.show()
    
    # 保存結果到 CSV
    results.to_csv(output_file, index=False)
    print(f"推論結果已保存為 '{output_file}'")
    
    return results, chain_summary

# 主程式 - 推論測試
if __name__ == "__main__":
    # 設置隨機種子以確保可重現性
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 假設新數據文件路徑
    new_data_file = 'new_attack_data.jsonl'  # 替換為你的新數據文件路徑
    
    # 步驟 1: 載入模型和預處理器
    model, tokenizer, scaler = load_model_and_preprocessors()
    
    # 步驟 2: 載入新數據
    new_df = load_new_jsonl_file(new_data_file, chain_instance_id="test_chain_001")
    
    # 步驟 3: 預處理新數據
    X_seq_new, X_static_new, processed_df = preprocess_new_data(
        new_df, tokenizer, scaler, max_sequence_length=40
    )
    
    # 步驟 4: 進行推論並保存結果
    results, chain_summary = perform_inference(
        model, X_seq_new, X_static_new, processed_df, output_file='inference_results.csv'
    )
    
    print("\n推論完成！")
    print("="*50)
    print("生成的文件：")
    print("1. inference_results.csv - 詳細推論結果")
    print("2. confidence_distribution.png - 預測信心值分佈圖")
    print("="*50)
