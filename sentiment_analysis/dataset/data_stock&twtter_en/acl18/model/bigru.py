# === 配置参数 ===
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# === 获取当前脚本名作为保存路径 ===
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]  # e.g., bigru
SAVE_DIR = os.path.join('result', SCRIPT_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

# === 参数设定 ===
BASE_DIR = 'data/combined_data'
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAPS = [1, 3, 6, 10, 15]  # 多步预测

# === 指标函数 ===
def calc_metrics(preds, targets):
    mae = torch.mean(torch.abs(preds - targets)).item()
    mse = torch.mean((preds - targets) ** 2).item()
    mape = torch.mean(torch.abs((preds - targets) / (targets + 1e-8))).item()
    tic_numerator = torch.sum((preds - targets) ** 2)
    tic_denominator = torch.sum((preds ** 2 + targets ** 2))
    tic = (tic_numerator / tic_denominator).item()
    ia_numerator = torch.sum((preds - targets) ** 2)
    ia_denominator = torch.sum((torch.abs(preds - torch.mean(targets)) + torch.abs(targets - torch.mean(targets))) ** 2)
    ia = 1 - ia_numerator / (ia_denominator + 1e-8)
    return {'mae': mae, 'mse': mse, 'mape': mape, 'tic': tic, 'ia': ia}

# === 自定义Dataset ===
class StockDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.samples = []
        for i in range(len(features) - seq_len):
            x = features[i:i+seq_len]
            y = labels[i+seq_len]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === 模型定义 ===
class BiGRU(nn.Module):
    def __init__(self, input_dim, pred_gap, hidden_dim=64, num_layers=2):
        super().__init__()
        self.bigru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, pred_gap)  # 双向GRU → 输出维度乘2

    def forward(self, x):
        out, _ = self.bigru(x)
        out = out[:, -1, :]  # 取序列最后一个时间步的双向拼接输出
        return self.fc(out)

# === 遍历多个特征子文件夹 ===
subfolders = ['10features', '15features', '20features', '25features', '30features']

for folder in tqdm(subfolders, desc="子文件夹遍历进度"):
    DATA_DIR = os.path.join(BASE_DIR, folder)
    FEATURE_COLS = list(range(1, 1 + int(folder.replace('features', ''))))
    all_results = {}

    for PRED_GAP in tqdm(GAPS, desc=f"{folder} 步长迭代", leave=False):
        all_features, all_labels = [], []
        scaler = MinMaxScaler()

        for fname in os.listdir(DATA_DIR):
            if not fname.endswith('.csv'):
                continue
            fpath = os.path.join(DATA_DIR, fname)
            df = pd.read_csv(fpath).dropna()
            if len(df) < SEQ_LEN + PRED_GAP + 1:
                continue
            features = df.iloc[:, FEATURE_COLS].values
            labels = [df['n_close'].shift(-i).values for i in range(PRED_GAP)]
            labels = np.stack(labels, axis=1)[:-PRED_GAP]
            features = features[:-PRED_GAP]
            features = scaler.fit_transform(features)
            all_features.append(features)
            all_labels.append(labels)

        if len(all_features) == 0:
            print(f"跳过 {folder} G{PRED_GAP}，无有效数据")
            continue

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        dataset = StockDataset(all_features, all_labels, SEQ_LEN)

        train_size = int(0.8 * len(dataset))
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model = BiGRU(input_dim=len(FEATURE_COLS), pred_gap=PRED_GAP).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            print(f"{folder} G{PRED_GAP} Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_dataset):.4f}")

        # === 评估 ===
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calc_metrics(all_preds, all_targets)
        all_results[f"G{PRED_GAP}"] = metrics

        # === 保存模型参数 ===
        model_path = os.path.join(SAVE_DIR, f"model_{folder}_gap{PRED_GAP}.pth")
        torch.save(model.state_dict(), model_path)

    # === 保存CSV结果 ===
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df.index = [int(g[1:]) for g in results_df.index]
    results_df = results_df.sort_index()
    csv_path = os.path.join(SAVE_DIR, f"{folder}_result.csv")
    results_df.to_csv(csv_path)

    print(f"保存 {csv_path} 完成")
