# 基于LSTM的Wordle玩家表现预测

## 项目简介

本项目是基于深度学习（LSTM）模型对Wordle游戏玩家表现进行预测的期末作业。项目使用2023年MCM Problem C的数据，通过构建时间序列模型来预测玩家在游戏中的平均尝试次数。

## 项目结构

```
.
├── main.py                             # 主程序，整合所有步骤
├── step1_data_cleaning.py              # 步骤1: 数据清洗
├── step2_feature_engineering.py        # 步骤2: 特征工程
├── step3_build_model.py                # 步骤3: 构建LSTM模型
├── step4_train_model.py                # 步骤4: 训练模型
├── step5_evaluate_model.py             # 步骤5: 模型评估
├── step6_visualization.py              # 步骤6: 可视化结果
├── requirements.txt                    # 项目依赖
├── README.md                           # 项目说明文档
├── data                                # 数据文件（2023_MCM_Problem_C_Data.xlsx为原数据集）
├── feature                             # 特征工程文件
├── model                               # 模型生成文件
└── result                              # 结果文件
```

## 环境要求

- Python 3.8+
- 推荐使用虚拟环境

## 安装步骤

1. 克隆或下载项目到本地

2. 创建虚拟环境（推荐）:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. 安装依赖包:
```bash
pip install -r requirements.txt
```

## 使用方法

### 方法1: 运行完整流程（推荐）

运行主程序，将自动执行所有步骤：

```bash
python main.py
```

### 方法2: 逐步运行

如果只想运行特定步骤，可以单独执行：

```bash
# 步骤1: 数据清洗
python step1_data_cleaning.py

# 步骤2: 特征工程
python step2_feature_engineering.py

# 步骤3: 构建模型
python step3_build_model.py

# 步骤4: 训练模型
python step4_train_model.py

# 步骤5: 评估模型
python step5_evaluate_model.py

# 步骤6: 可视化结果
python step6_visualization.py
```

## 各步骤说明

### 步骤1: 数据清洗 (`step1_data_cleaning.py`)
- 处理缺失值
- 数据类型转换
- 计算目标变量（平均尝试次数、成功率）
- 保存清洗后的数据到 `data_cleaned.csv`

### 步骤2: 特征工程 (`step2_feature_engineering.py`)
- 创建时间序列特征（滞后特征、移动平均）
- 提取单词特征（长度、元音比例等）
- 数据标准化
- 构建LSTM输入序列
- 划分训练集和测试集
- 保存处理后的数据

### 步骤3: 构建模型 (`step3_build_model.py`)
- 定义LSTM模型架构
- 定义GRU模型（对比用）
- 定义双向LSTM模型（对比用）
- 测试模型结构
- 保存模型配置

### 步骤4: 训练模型 (`step4_train_model.py`)
- 加载训练数据
- 定义损失函数和优化器
- 训练LSTM模型
- 实现早停机制
- 保存训练好的模型和训练历史

### 步骤5: 模型评估 (`step5_evaluate_model.py`)
- 加载训练好的模型
- 在测试集上进行预测
- 计算评估指标（MAE, RMSE, R², MAPE）
- 生成预测对比图
- 保存评估结果

### 步骤6: 可视化结果 (`step6_visualization.py`)
- 生成时间序列预测图
- 生成性能对比图
- 生成数据特征分析图
- 生成误差分析图
- 生成总结报告图


## 评估指标

模型使用以下指标进行评估：
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **MSE (Mean Squared Error)**: 均方误差
- **RMSE (Root Mean Squared Error)**: 均方根误差
- **R² (R-squared)**: 决定系数
- **MAPE (Mean Absolute Percentage Error)**: 平均绝对百分比误差

## 输出文件说明

运行完成后，将生成以下文件：

### 数据文件（./data）
- `data_cleaned.csv`: 清洗后的数据

### 特征文件（./feature）
- `X_train.npy`, `X_test.npy`: 训练和测试特征数据
- `y_train.npy`, `y_test.npy`: 训练和测试标签数据
- `scaler_X.pkl`, `scaler_y.pkl`: 特征和目标变量的标准化器
- `feature_info.json`: 特征信息

### 模型文件（./model）
- `model_lstm.pth`: 训练好的LSTM模型
- `model_config.json`: 模型配置文件

### 评估结果（./result）
- `evaluation_results.json`: 评估结果（JSON格式）
- `model_analysis_report.png`: 可视化评估


## 项目特点

1. **模块化设计**: 每个步骤独立运行，便于调试和修改
2. **完整流程**: 从数据探索到结果可视化的完整机器学习流程
3. **时间序列建模**: 使用LSTM捕捉时间序列中的长期依赖关系
4. **特征工程**: 包含滞后特征、移动平均、单词特征等多源特征
5. **可视化丰富**: 提供多种可视化图表分析模型性能

## 注意事项

1. **数据文件**: 确保 `2023_MCM_Problem_C_Data.xlsx` 文件在项目目录
2. **GPU加速**: 如果安装CUDA版本的PyTorch，训练速度会更快
3. **训练时间**: 模型训练可能需要几分钟到十几分钟，取决于硬件配置
4. **内存需求**: 确保有足够的内存（建议至少4GB）

## 可能的问题与解决

### 问题1: 无法读取Excel文件
**解决**: 确保安装了 `openpyxl`: `pip install openpyxl`

### 问题2: 中文显示乱码
**解决**: 确保系统安装了中文字体，matplotlib会自动使用SimHei或Microsoft YaHei

### 问题3: CUDA相关错误
**解决**: 如果没有GPU，PyTorch会自动使用CPU，不影响运行

### 问题4: 内存不足
**解决**: 可以减小batch_size（在step4中修改）或减小seq_length（在step2中修改）

## 未来改进方向

1. **模型扩展**: 尝试Transformer、Temporal Fusion Transformer等更先进的模型
2. **特征扩展**: 引入更多外部特征（如天气、节假日等）
3. **超参数优化**: 使用网格搜索或贝叶斯优化自动调参
4. **集成学习**: 使用多个模型的集成预测
5. **多步预测**: 扩展为多步时间序列预测任务

## 参考文献

- 2023 MCM Problem C: Wordle Problem
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

## 作者信息

本项目为大数据系统原理与应用课程期末作业。

## 许可

本项目仅用于学习和研究目的。
