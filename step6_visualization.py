"""
步骤6: 可视化结果分析
目标: 生成各种可视化图表，分析模型性能和预测结果
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# 设置中文字体
import matplotlib.pyplot as plt

# 尝试手动指定一个系统自带的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# 启用数学公式的内置解析器
plt.rcParams['mathtext.fontset'] = 'stix'

print("=" * 60)
print("步骤6: 可视化结果分析 ")
print("=" * 60)

# 1. 加载数据
print("\n加载数据...")
if not os.path.exists('./result/evaluation_results.json'):
    print("错误: 未找到 evaluation_results.json，请先运行 step6")
    exit(1)

with open('./result/evaluation_results.json', 'r', encoding='utf-8') as f:
    eval_results = json.load(f)

# 修键名映射
y_train_true = np.array(eval_results['y_train_true'])
y_train_pred = np.array(eval_results['train_pred'])
y_test_true = np.array(eval_results['y_test_true'])
y_test_pred = np.array(eval_results['test_pred'])

# 2. 创建综合分析看板
print("正在生成可视化图表...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

# --- 子图1: 测试集预测结果对比 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(y_test_true, label='实际平均尝试次数', color='#1f77b4', linewidth=2, marker='o', markersize=4, alpha=0.7)
ax1.plot(y_test_pred, label='模型预测值', color='#d62728', linewidth=2, linestyle='--')
ax1.set_title('测试集：预测值 vs 实际值 (波动分析)', fontsize=14)
ax1.set_xlabel('样本编号 (按时间顺序)')
ax1.set_ylabel('尝试次数')
ax1.legend()

# --- 子图2: 预测残差分析 (Residuals) ---
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test_true - y_test_pred
sns.histplot(residuals, kde=True, ax=ax2, color='purple')
ax2.axvline(0, color='black', linestyle='--')
ax2.set_title('预测误差分布 (残差分析)', fontsize=14)
ax2.set_xlabel('误差 (实际值 - 预测值)')

# --- 子图3: 拟合优度散点图 ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_test_true, y_test_pred, alpha=0.6, color='green')
# 画出 y=x 参考线
lims = [min(y_test_true.min(), y_test_pred.min()), max(y_test_true.max(), y_test_pred.max())]
ax3.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
ax3.set_title(f'拟合优度 ($R^2$: {eval_results["test_metrics"]["R2"]:.3f})', fontsize=14)
ax3.set_xlabel('实际值')
ax3.set_ylabel('预测值')

# --- 子图4: 特征相关性热力图  ---
# 读取含NLP特征的临时数据（为了展示单词属性的影响）
ax4 = fig.add_subplot(gs[1, 1])
if os.path.exists('./data/data_cleaned.csv'):
    # 这里模拟提取部分NLP特征的相关性，增加论文的可解释性
    df_temp = pd.read_csv('./data/data_cleaned.csv')
    # 选取模型用到的核心特征
    corr_cols = ['mean_tries', 'total_reports', 'hard_mode_count', 'success_rate']
    corr_matrix = df_temp[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', ax=ax4)
    ax4.set_title('核心特征相关性矩阵', fontsize=14)
else:
    ax4.text(0.5, 0.5, '需要 data_cleaned.csv 生成热力图', ha='center')

plt.suptitle('Wordle 玩家表现预测模型分析报告', fontsize=20, fontweight='bold', y=0.95)
plt.savefig('./result/model_analysis_report.png', dpi=300, bbox_inches='tight')
print("\n可视化报告已保存至: model_analysis_report.png")
plt.show()
