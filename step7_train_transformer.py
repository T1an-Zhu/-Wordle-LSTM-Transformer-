"""
步骤7: Transformer 模型训练
目标: 与lstm模型进行对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import math
import pickle

# 1. Transformer 相关类定义 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class WordleTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :] # 取最后一个时间步
        return self.fc(x)

class WordleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# 2. 训练主逻辑 
def train():
    print("="*60)
    print("步骤7: Transformer 模型训练")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置
    FEATURE_DIR = 'feature'
    MODEL_DIR = 'model'
    
    # 加载特征信息
    with open(os.path.join(FEATURE_DIR, 'feature_info.json'), 'r') as f:
        info = json.load(f)

    # 加载数据
    X_train = np.load(os.path.join(FEATURE_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(FEATURE_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(FEATURE_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(FEATURE_DIR, 'y_test.npy'))

    train_loader = DataLoader(WordleDataset(X_train, y_train), batch_size=16, shuffle=True)

    # 初始化模型 (注意：d_model 必须能被 nhead 整除)
    model = WordleTransformer(input_size=info['input_size'], d_model=64, nhead=4).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    print(f"开始训练 | 设备: {device}")

    for epoch in range(300):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(torch.FloatTensor(X_test).to(device))
            test_loss = criterion(test_outputs, torch.FloatTensor(y_test).to(device)).item()
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)

        scheduler.step(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'model_transformer.pth'))
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | Test Loss: {test_loss:.6f}")
    
    history = {
        'train_loss': train_losses,
        'test_loss': test_losses
    }
    history_path = os.path.join(MODEL_DIR, 'transformer_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"训练完成，最佳权重已保存至 {MODEL_DIR}/model_transformer.pth")

if __name__ == "__main__":
    train()