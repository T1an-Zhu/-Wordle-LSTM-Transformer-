"""
步骤8: LSTM vs Transformer 横向对比
目标: 比较两种模型在测试集上的表现，并生成对比图表
"""
import torch
import numpy as np
import json
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 从 step3 导入 LSTM 模型定义
from step3_build_model import WordleLSTMModel
# 从 step7 导入 Transformer 模型定义
from step7_train_transformer import WordleTransformer

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def compare_models():
    print("="*60)
    print("步骤8: LSTM vs Transformer 横向对比")
    print("="*60)
    
    # 路径设置
    FEATURE_DIR = 'feature'
    MODEL_DIR = 'model'
    RESULT_DIR = 'result'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据和归一化器
    X_test = np.load(os.path.join(FEATURE_DIR, 'X_test.npy'))
    y_test_scaled = np.load(os.path.join(FEATURE_DIR, 'y_test.npy'))
    with open(os.path.join(FEATURE_DIR, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_config.json'), 'r') as f:
        lstm_cfg = json.load(f)
    
    y_true = scaler_y.inverse_transform(y_test_scaled).flatten()
    X_tensor = torch.FloatTensor(X_test).to(device)

    # 2. 加载 LSTM 模型
    lstm_model = WordleLSTMModel(
        input_size=lstm_cfg['input_size'],
        hidden_size=48,   
        num_layers=1      
    ).to(device)

    lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model_lstm.pth'), map_location=device))
    lstm_model.eval()

    # 3. 加载 Transformer 模型
    trans_model = WordleTransformer(input_size=lstm_cfg['input_size']).to(device)
    trans_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model_transformer.pth')))
    trans_model.eval()

    # 4. 执行预测
    with torch.no_grad():
        y_pred_lstm = scaler_y.inverse_transform(lstm_model(X_tensor).cpu().numpy()).flatten()
        y_pred_trans = scaler_y.inverse_transform(trans_model(X_tensor).cpu().numpy()).flatten()

    # 5. 计算指标
    def get_metrics(true, pred):
        return {
            'R2': r2_score(true, pred),
            'MAE': mean_absolute_error(true, pred),
            'RMSE': np.sqrt(mean_squared_error(true, pred))
        }

    m_lstm = get_metrics(y_true, y_pred_lstm)
    m_trans = get_metrics(y_true, y_pred_trans)

    # 打印对比表
    print(f"\n{'指标':<10} | {'LSTM':<12} | {'Transformer':<12}")
    print("-" * 40)
    for k in ['R2', 'MAE', 'RMSE']:
        print(f"{k:<10} | {m_lstm[k]:<12.4f} | {m_trans[k]:<12.4f}")

    # 6. 可视化
    plt.figure(figsize=(14, 7))
    
    # 预测曲线对比
    plt.subplot(1, 2, 1)
    samples = 60 # 选取前60个测试样本显示
    plt.plot(y_true[:samples], 'k--', label='实际值', alpha=0.6)
    plt.plot(y_pred_lstm[:samples], 'b-', label=f'LSTM (R²={m_lstm["R2"]:.2f})')
    plt.plot(y_pred_trans[:samples], 'r-', label=f'Transformer (R²={m_trans["R2"]:.2f})')
    plt.title('预测曲线对比 (前60个测试样本)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 指标条形图对比
    plt.subplot(1, 2, 2)
    labels = ['R2 Score', 'MAE', 'RMSE']
    lstm_vals = [m_lstm['R2'], m_lstm['MAE'], m_lstm['RMSE']]
    trans_vals = [m_trans['R2'], m_trans['MAE'], m_trans['RMSE']]
    
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, lstm_vals, width, label='LSTM', color='skyblue')
    plt.bar(x + width/2, trans_vals, width, label='Transformer', color='salmon')
    plt.xticks(x, labels)
    plt.title('模型指标横向对比')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'model_comparison.png'))
    print(f"\n对比图表已保存至 {RESULT_DIR}/model_comparison.png")
    plt.show()

def plot_loss_curves():
    # 1. 读取保存好的历史数据
    with open('model/lstm_history.json', 'r') as f:
        lstm_h = json.load(f)
    with open('model/transformer_history.json', 'r') as f:
        trans_h = json.load(f)

    # 2. 创建画布
    plt.figure(figsize=(12, 6))

    # 左图：LSTM 的收敛过程
    plt.subplot(1, 2, 1)
    plt.plot(lstm_h['train_loss'], label='训练 Loss (LSTM)', color='#1f77b4')
    plt.plot(lstm_h['test_loss'], label='测试 Loss (LSTM)', color='#ff7f0e', linestyle='--')
    plt.title('LSTM 训练收敛曲线')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：Transformer 的收敛过程
    plt.subplot(1, 2, 2)
    plt.plot(trans_h['train_loss'], label='训练 Loss (Transformer)', color='#2ca02c')
    plt.plot(trans_h['test_loss'], label='测试 Loss (Transformer)', color='#d62728', linestyle='--')
    plt.title('Transformer 训练收敛曲线')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # 3. 保存到结果文件夹
    plt.savefig('result/loss_comparison.png', dpi=300)
    print("Loss 对比曲线已保存至 result/loss_comparison.png")

if __name__ == "__main__":
    compare_models()
    plot_loss_curves()