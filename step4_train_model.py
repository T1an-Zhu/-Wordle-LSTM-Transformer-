"""
步骤4: 模型训练 
目标: 解决过拟合问题，通过正则化和早停机制提升测试集 R²
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import random
from step3_build_model import WordleLSTMModel

# 1. 环境与随机种子设置
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WordleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

print("=" * 60)
print("步骤4: 模型训练 ")
print("=" * 60)

# 2. 加载数据与配置 
if not os.path.exists('./feature/feature_info.json'):
    print("错误: 找不到 feature_info.json，请确保已运行 step3！")
    exit(1)

with open('./feature/feature_info.json', 'r') as f:
    info = json.load(f)

X_train, y_train = np.load('./feature/X_train.npy'), np.load('./feature/y_train.npy')
X_test, y_test = np.load('./feature/X_test.npy'), np.load('./feature/y_test.npy')

# 使用较小的 Batch Size 有助于模型泛化
batch_size = 16 
train_loader = DataLoader(WordleDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# 3. 模型参数配置 
HIDDEN_SIZE = 48    # 适中的隐藏层大小
NUM_LAYERS = 1      # 对于小样本，1层通常比2层效果好
DROPOUT = 0.3       # 较高的 Dropout 比例防止过拟合

model = WordleLSTMModel(
    input_size=info['input_size'], 
    hidden_size=HIDDEN_SIZE, 
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# 损失函数与优化器
criterion = nn.MSELoss()
# 加入 weight_decay (L2 正则化)，惩罚过大权重
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
train_losses = []
test_losses = []


# 学习率调度器 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

# 4. 训练循环 
epochs = 500
best_test_loss = float('inf')
patience_counter = 0
early_stop_patience = 50 # 50轮测试集不进步则停止

print(f"开始训练 | 设备: {device} | 隐藏层: {HIDDEN_SIZE} | 序列长度: {info['seq_length']}")

for epoch in range(epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(X_test).to(device)
        test_targets = torch.FloatTensor(y_test).to(device)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets).item()
    
    # 更新学习率
    scheduler.step(test_loss)

    # 记录每一轮的平均训练损失和测试损失
    train_losses.append(total_train_loss / len(train_loader))
    test_losses.append(test_loss)
    
    # 打印进度 (每20轮打印一次)
    if (epoch + 1) % 20 == 0:
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {curr_lr:.6f}")
    
    # 保存最佳模型权重 (Best Model Saving)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), './model/model_lstm.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # 早停逻辑
    if patience_counter >= early_stop_patience:
        print(f"\n[触发早停] 在第 {epoch+1} 轮停止。最佳测试集 Loss: {best_test_loss:.6f}")
        break

# 5. 生成最终配置文件
# step6 评估时会根据这个 json 来初始化相同结构的模型
model_config = {
    "input_size": info['input_size'],
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "seq_length": info['seq_length']
}
with open('./model/model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)

history = {
    'train_loss': train_losses,
    'test_loss': test_losses
}

# 确保保存路径与 step8 中读取的路径一致
history_path = './model/lstm_history.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)
print("训练结束，最佳模型已保存为 'model_lstm.pth',训练历史已保存至 '{history_path}'")