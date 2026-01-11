"""
步骤2: 特征工程
目标: 构建时间序列特征，为LSTM模型准备数据
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import json

print("=" * 60)
print("步骤2: 增强型特征工程")
print("=" * 60)

# 1. 读取数据
if not os.path.exists('./data/data_cleaned.csv'):
    print("错误: 未找到 data_cleaned.csv，请先运行 step1")
    exit(1)

df = pd.read_csv('./data/data_cleaned.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"原始数据读取成功，共 {len(df)} 行")

# 2. 计算单词 NLP 特征
def get_word_features(word):
    word = str(word).lower()
    vowels = "aeiou"
    return pd.Series({
        'unique_chars': len(set(word)),
        'vowel_count': sum(1 for c in word if c in vowels),
        'has_duplicate': 1 if len(set(word)) < 5 else 0,
        'char_diversity': len(set(word)) / 5
    })

print("正在计算单词语言学特征...")
df_nlp = df['word'].apply(get_word_features)
df = pd.concat([df, df_nlp], axis=1)

# 3. 添加滞后特征 (Lag Feature)
# 这一步会产生1个NaN（第一行没昨天），后面会处理
df['lag_1'] = df['mean_tries'].shift(1)

# 4. 时间特征
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# 5. 特征选择
# 确保这些列在 data_cleaned.csv 中确实存在
feature_columns = [
    'total_reports', 'hard_mode_count', 'month', 'day_of_week',
    'unique_chars', 'vowel_count', 'has_duplicate', 'char_diversity', 
    'success_rate', 'lag_1'
]
target_column = 'mean_tries'

# 检查缺失值情况
print("\n检查各特征列的缺失值数量:")
print(df[feature_columns + [target_column]].isnull().sum())

# 删除含有缺失值的行（主要是lag_1产生的第1行）
df_final = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

if len(df_final) == 0:
    print("!!! 严重错误: 处理后数据集变为空白。请检查 data_cleaned.csv 的列名是否正确 !!!")
    print("当前列名列表:", df.columns.tolist())
    exit(1)

print(f"清理完成，可用样本量: {len(df_final)}")

# 6. 数据归一化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df_final[feature_columns])
y_scaled = scaler_y.fit_transform(df_final[[target_column]])

# 7. 构建序列 (seq_length=7)
seq_length = 7 
X_seq, y_seq = [], []

# i:i+7 预测 i+6 (预测当前序列的最后一天)
for i in range(len(df_final) - seq_length + 1):
    X_seq.append(X_scaled[i : i + seq_length])
    # 目标值 y 是这个 7 天序列中最后一天的结果
    y_seq.append(y_scaled[i + seq_length - 1]) 

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# 8. 划分训练/测试集
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# 9. 保存
np.save('./feature/X_train.npy', X_train)
np.save('./feature/X_test.npy', X_test)
np.save('./feature/y_train.npy', y_train)
np.save('./feature/y_test.npy', y_test)

with open('./feature/scaler_X.pkl', 'wb') as f: pickle.dump(scaler_X, f)
with open('./feature/scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)

# 保存模型需要的元数据
with open('./feature/feature_info.json', 'w') as f:
    json.dump({
        'input_size': len(feature_columns), 
        'seq_length': seq_length,
        'feature_columns': feature_columns
    }, f)

print(f"\n[完成] 特征工程结束。")
print(f"训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")