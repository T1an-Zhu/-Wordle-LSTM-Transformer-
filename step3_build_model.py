"""
步骤3: 构建LSTM模型
目标: 定义和构建用于Wordle玩家表现预测的LSTM模型
"""

import torch
import torch.nn as nn
import json
import os

print("=" * 60)
print("步骤3: 构建优化版LSTM模型")
print("=" * 60)

class WordleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(WordleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层：处理时序特征
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 全连接层：增加非线性表达，帮助预测剧烈波动
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步，送入全连接层
        out = self.fc(out[:, -1, :])
        return out

# 加载配置并生成模型保存信息
if os.path.exists('./feature/feature_info.json'):
    with open('./feature/feature_info.json', 'r') as f:
        info = json.load(f)
    
    model_config = {
        "input_size": info['input_size'],
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_length": info['seq_length']
    }
    
    with open('./model/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=4)
    print("模型配置已生成。")