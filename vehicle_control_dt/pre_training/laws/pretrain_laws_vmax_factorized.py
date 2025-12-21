#!/usr/bin/env python3
import argparse, os, math
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

class CSVDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.x_cols = ["radius_m","mu","curvature_1pm"]  # ★ g/safetyは学習から外す
        X = df[self.x_cols].to_numpy(np.float32)
        denom = np.sqrt(df["g_mps2"].to_numpy(np.float32)) * df["safety_margin"].to_numpy(np.float32)
        y = (df["vmax_mps"].to_numpy(np.float32) / np.maximum(denom, 1e-6)).reshape(-1,1)  # v_base ≈ √(μr)
        self.X, self.y = X, y
        self.x_mean = self.X.mean(0, keepdims=True).astype(np.float32)
        self.x_std  = np.maximum(self.X.std(0, keepdims=True).astype(np.float32), 1e-3)  # クリップ
        self.y_mean = np.float32(self.y.mean())
        self.y_std  = np.float32(max(float(self.y.std()), 1e-3))

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", default="models/vmax_factorized.pt")
    ap.add_argument("--save_scaler", default="models/scaler_factorized.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    ds_full = CSVDataset(args.csv)
    n = len(ds_full); n_tr = int(0.9*n); n_va = n - n_tr
    ds_tr, ds_va = random_split(ds_full, [n_tr, n_va], generator=torch.Generator().manual_seed(42))

    x_mean, x_std = ds_full.x_mean, ds_full.x_std
    y_mean, y_std = ds_full.y_mean, ds_full.y_std
    def nx(x): return (x - x_mean) / x_std
    def dny(y): return y * y_std + y_mean

    tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True)
    va = DataLoader(ds_va, batch_size=args.bs, shuffle=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=len(ds_full.x_cols)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss = nn.MSELoss()

    best = 1e9
    for e in range(1, args.epochs+1):
        model.train(); tl = 0.0
        for X,Y in tr:
            Xn = torch.from_numpy(nx(X.numpy())).to(dev)
            Yn = torch.from_numpy(((Y.numpy()-y_mean)/y_std)).to(dev)
            opt.zero_grad(); L = loss(model(Xn), Yn); L.backward(); opt.step()
            tl += L.item() * len(X)
        tl /= len(ds_tr)

        model.eval(); vmse=0.0; vmae=0.0
        with torch.no_grad():
            for X,Y in va:
                Xn = torch.from_numpy(nx(X.numpy())).to(dev)
                yhat_n = model(Xn).cpu().numpy()
                yhat = dny(yhat_n)  # v_base に戻す
                vmse += ((yhat - Y.numpy())**2).mean() * len(X)
                vmae += (np.abs(yhat - Y.numpy())).mean() * len(X)
        vmse/=len(ds_va); vmae/=len(ds_va)
        print(f"[{e:03d}] train_mse={tl:.6f}  val_mse={vmse:.6f}  val_mae(base)={vmae:.6f}")
        if vmse < best:
            best = vmse
            torch.save(model.state_dict(), args.save)
            np.savez(args.save_scaler,
                     x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                     x_cols=np.array(ds_full.x_cols))
            print(f"  -> saved {args.save}  (scaler: {args.save_scaler})")

if __name__=="__main__":
    main()
