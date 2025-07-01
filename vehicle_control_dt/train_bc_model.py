import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# === データセット定義 ===
class ExpertDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df[["target_wp_x", "target_wp_y", "pos_x", "pos_y", "yaw", "velocity",
                     "perp_error", "heading_error", "passed"]].values.astype("float32")
        self.y = df[["steer_angle", "throttle"]].values.astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === モデル定義 ===
class ControlMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === ハイパーパラメータ ===
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

# === データ読み込み ===
dataset = ExpertDataset("expert_data/expert_data.csv")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === モデル構築（継続学習対応） ===
model = ControlMLP()
model_path = "models/bc_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"✅ Pretrained model loaded from: {model_path}")
else:
    print("🆕 No pretrained model found. Training from scratch.")

# === 学習ループ ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x_batch)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}")

# === 保存 ===
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"💾 Model saved to: {model_path}")
