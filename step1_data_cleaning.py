"""
步骤1: 数据清洗和预处理
目标: 清理数据，处理缺失值，转换数据类型，为后续分析做准备
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("步骤1: 数据清洗和预处理")
print("=" * 60)

# 读取数据
data_file = './data/2023_MCM_Problem_C_Data.xlsx'
print(f"\n正在读取文件: {data_file}")

df_raw = pd.read_excel(data_file)

# 设置正确的列名（第一行是标题）
new_columns = df_raw.iloc[0].values
df = df_raw.iloc[1:].copy()
df.columns = new_columns
df = df.reset_index(drop=True)

print(f"\n原始数据形状: {df.shape}")
print(f"原始列名: {df.columns.tolist()}")

# 重命名列以便后续处理
column_mapping = {
    'Date': 'date',
    'Contest number': 'contest_number',
    'Word': 'word',
    'Number of  reported results': 'total_reports',
    'Number in hard mode': 'hard_mode_count',
    'Percent in': 'percent_in',
    '1 try': 'tries_1',
    '2 tries': 'tries_2',
    '3 tries': 'tries_3',
    '4 tries': 'tries_4',
    '5 tries': 'tries_5',
    '6 tries': 'tries_6',
    '7 or more tries (X)': 'tries_7plus'
}

# 只重命名存在的列
df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

print(f"\n重命名后的列名: {df.columns.tolist()}")

# 1. 处理日期列
print("\n处理日期列...")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(f"日期转换后，缺失值数量: {df['date'].isna().sum()}")

# 2. 处理数值列（尝试次数分布）
numeric_columns = ['contest_number', 'total_reports', 'hard_mode_count', 
                   'tries_1', 'tries_2', 'tries_3', 'tries_4', 
                   'tries_5', 'tries_6', 'tries_7plus']

for col in numeric_columns:
    if col in df.columns:
        # 转换为数值类型，无法转换的设为NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"{col}: 转换后缺失值数量 = {df[col].isna().sum()}")

# 3. 处理缺失值
print("\n检查缺失值...")
print(df.isnull().sum())

# 删除日期为空的记录（无效记录）
initial_rows = len(df)
df = df.dropna(subset=['date'])
print(f"\n删除日期缺失的记录: {initial_rows - len(df)} 行")

# 4. 按日期排序
df = df.sort_values('date').reset_index(drop=True)

# 5. 计算目标变量：平均尝试次数
# 基于各尝试次数的分布，计算加权平均尝试次数
def calculate_mean_tries(row):
    """计算平均尝试次数"""
    total = 0
    count = 0
    tries_cols = ['tries_1', 'tries_2', 'tries_3', 'tries_4', 'tries_5', 'tries_6', 'tries_7plus']
    try_values = [1, 2, 3, 4, 5, 6, 7]  # 7次及以上按7次计算
    
    for i, col in enumerate(tries_cols):
        if col in row and pd.notna(row[col]):
            num_players = row[col]
            total += num_players * try_values[i]
            count += num_players
    
    if count > 0:
        return total / count
    else:
        return np.nan

print("\n计算平均尝试次数...")
df['mean_tries'] = df.apply(calculate_mean_tries, axis=1)
print(f"平均尝试次数统计:")
print(df['mean_tries'].describe())

# 6. 计算成功率（1-6次尝试成功视为成功）
def calculate_success_rate(row):
    """计算成功率"""
    tries_cols = ['tries_1', 'tries_2', 'tries_3', 'tries_4', 'tries_5', 'tries_6']
    success_count = 0
    total_count = 0
    
    for col in tries_cols:
        if col in row and pd.notna(row[col]):
            success_count += row[col]
        if col in row and pd.notna(row[col]):
            total_count += row[col]
    
    # 加上失败数（7次及以上）
    if 'tries_7plus' in row and pd.notna(row['tries_7plus']):
        total_count += row['tries_7plus']
    
    if total_count > 0:
        return success_count / total_count
    else:
        return np.nan

print("\n计算成功率...")
df['success_rate'] = df.apply(calculate_success_rate, axis=1)
print(f"成功率统计:")
print(df['success_rate'].describe())

# 7. 保存清洗后的数据
output_file = './data/data_cleaned.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n清洗后的数据已保存到: {output_file}")
print(f"最终数据形状: {df.shape}")

# 8. 数据摘要
print("\n" + "=" * 60)
print("数据清洗摘要")
print("=" * 60)
print(f"总记录数: {len(df)}")
print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
print(f"平均尝试次数范围: {df['mean_tries'].min():.2f} - {df['mean_tries'].max():.2f}")
print(f"成功率范围: {df['success_rate'].min():.2%} - {df['success_rate'].max():.2%}")

# 显示前几行清洗后的数据
print("\n清洗后的数据预览（前5行）:")
print(df[['date', 'word', 'mean_tries', 'success_rate', 'total_reports']].head())
