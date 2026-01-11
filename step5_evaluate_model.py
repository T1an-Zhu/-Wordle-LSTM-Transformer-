"""
步骤5: 模型评估和预测
目标: 评估训练好的模型，计算各种评估指标，进行预测
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("步骤5: 模型评估和预测 ")
print("=" * 60)

# 1. 加载数据和配置
print("\n 1.加载数据与模型配置...")
X_train = np.load('./feature/X_train.npy')
X_test = np.load('./feature/X_test.npy')
y_train_scaled = np.load('./feature/y_train.npy')
y_test_scaled = np.load('./feature/y_test.npy')

with open('./feature/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

with open('./model/model_config.json', 'r') as f:
    config = json.load(f)

# 2. 初始化模型并加载权重
print("\n 2.加载训练好的权重...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from step3_build_model import WordleLSTMModel

model = WordleLSTMModel(
    input_size=config['input_size'],
    hidden_size=config['hidden_size'],
    num_layers=config.get('num_layers', 2),
    dropout=config.get('dropout', 0.2)
).to(device)

if os.path.exists('./model/model_lstm.pth'):
    model.load_state_dict(torch.load('./model/model_lstm.pth', map_location=device))
    model.eval()
    print(f"成功加载模型权重，使用设备: {device}")
else:
    print("错误: 未找到 model_lstm.pth，请先运行 step5")
    exit(1)

# 3. 执行预测与反归一化
print("\n 3. 执行模型推理...")
with torch.no_grad():
    # 预测训练集
    train_pred_scaled = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
    # 预测测试集
    test_pred_scaled = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

# 将数据转回原始量级（平均尝试次数）
y_train_true = scaler_y.inverse_transform(y_train_scaled)
y_train_pred = scaler_y.inverse_transform(train_pred_scaled)
y_test_true = scaler_y.inverse_transform(y_test_scaled)
y_test_pred = scaler_y.inverse_transform(test_pred_scaled)

# 4. 计算详细评估指标
def get_metrics(true, pred):
    return {
        'MAE': float(mean_absolute_error(true, pred)),
        'MSE': float(mean_squared_error(true, pred)),
        'RMSE': float(np.sqrt(mean_squared_error(true, pred))),
        'R2': float(r2_score(true, pred)),
        'MAPE': float(np.mean(np.abs((true - pred) / true)) * 100)
    }

train_metrics = get_metrics(y_train_true, y_train_pred)
test_metrics = get_metrics(y_test_true, y_test_pred)

print("\n" + "="*30)
print(f"训练集 R²: {train_metrics['R2']:.4f}")
print(f"测试集 R²: {test_metrics['R2']:.4f}")
print(f"测试集 MAE: {test_metrics['MAE']:.4f}")
print("="*30)

# 5. 保存完整结果供 step7 绘图使用
# 必须包含 step7 期待的所有键名
evaluation_results = {
    'y_train_true': y_train_true.flatten().tolist(),
    'train_pred': y_train_pred.flatten().tolist(),
    'y_test_true': y_test_true.flatten().tolist(),
    'test_pred': y_test_pred.flatten().tolist(),
    'train_metrics': train_metrics,
    'test_metrics': test_metrics
}

with open('./result/evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

print("\n 4. 评估数据已补全并保存至 evaluation_results.json")
