import os
import glob
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    # project root execution
    from world_training.laws.world_mlp import WorldMLP
except ModuleNotFoundError:
    # laws folder direct execution
    from world_mlp import WorldMLP

OBS_DIM = 19
ACT_DIM = 2

# input = obs_norm + prev_action + action
INPUT_DIM = OBS_DIM + ACT_DIM + ACT_DIM

# output = delta_obs_norm
OUTPUT_DIM = OBS_DIM


class WorldDataset(Dataset):
    def __init__(self, data_dir):
        xs = []
        ys = []

        ds_dirs = sorted(glob.glob(os.path.join(data_dir, "ds_*")))

        if not ds_dirs:
            raise RuntimeError(f"No ds_* folders found under: {data_dir}")

        print("[LOAD DATASETS]")
        for ds_dir in ds_dirs:
            ds_name = os.path.basename(ds_dir)
            pkl_path = os.path.join(ds_dir, "trajectories_dt.pkl")

            if not os.path.exists(pkl_path):
                print(f"[WARN] skip missing: {pkl_path}")
                continue

            with open(pkl_path, "rb") as f:
                trajectories = pickle.load(f)

            print(f"{ds_name}: {len(trajectories)} trajectories")

            for traj_idx, item in enumerate(trajectories):
                obs = np.asarray(item["observations"], dtype=np.float32)
                act = np.asarray(item["actions"], dtype=np.float32)

                if obs.ndim != 2:
                    raise RuntimeError(f"[{ds_name}] observations must be 2D: {obs.shape}")

                if act.ndim != 2:
                    raise RuntimeError(f"[{ds_name}] actions must be 2D: {act.shape}")

                if obs.shape[1] != OBS_DIM:
                    raise RuntimeError(f"[{ds_name}] obs_dim mismatch: {obs.shape[1]} != {OBS_DIM}")

                if act.shape[1] != ACT_DIM:
                    raise RuntimeError(f"[{ds_name}] act_dim mismatch: {act.shape[1]} != {ACT_DIM}")

                if len(obs) != len(act):
                    raise RuntimeError(
                        f"[{ds_name}] length mismatch: obs={len(obs)}, act={len(act)}"
                    )

                T = obs.shape[0]
                if T < 2:
                    print(f"[WARN] skip short trajectory: {ds_name} traj={traj_idx} T={T}")
                    continue

                prev_act = np.zeros_like(act)
                prev_act[1:] = act[:-1]

                # t -> t+1
                x = np.concatenate(
                    [
                        obs[:-1],
                        prev_act[:-1],
                        act[:-1],
                    ],
                    axis=1,
                )

                y = obs[1:] - obs[:-1]

                if not np.isfinite(x).all():
                    raise RuntimeError(f"[{ds_name} traj={traj_idx}] x has NaN/Inf")

                if not np.isfinite(y).all():
                    raise RuntimeError(f"[{ds_name} traj={traj_idx}] y has NaN/Inf")

                xs.append(x)
                ys.append(y)

                print(
                    f"  traj={traj_idx} "
                    f"obs={obs.shape} act={act.shape} "
                    f"samples={len(x)}"
                )

        if not xs:
            raise RuntimeError("No training samples created.")

        self.x = np.concatenate(xs, axis=0).astype(np.float32)
        self.y = np.concatenate(ys, axis=0).astype(np.float32)

        print("\n[WorldDataset]")
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")

        if self.x.shape[1] != INPUT_DIM:
            raise RuntimeError(f"input dim mismatch: {self.x.shape[1]} != {INPUT_DIM}")

        if self.y.shape[1] != OUTPUT_DIM:
            raise RuntimeError(f"output dim mismatch: {self.y.shape[1]} != {OUTPUT_DIM}")

        self.print_target_stats()

    def print_target_stats(self):
        names = {
            6: "velocity",
            7: "perp_error",
            8: "heading_error",
            13: "v_ratio",
        }

        print("\n[target delta stats: normalized space]")
        for idx, name in names.items():
            v = self.y[:, idx]
            print(
                f"{idx:02d} {name:14s} "
                f"mean={v.mean():+.6f} "
                f"std={v.std():.6f} "
                f"min={v.min():+.6f} "
                f"max={v.max():+.6f}"
            )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loss_weights(device):
    weights = torch.ones(OUTPUT_DIM, device=device)

    # 重点学習対象
    weights[6] = 5.0    # velocity
    weights[7] = 2.0    # perp_error
    weights[8] = 2.0    # heading_error
    weights[13] = 5.0   # v_ratio

    return weights


def weighted_mse_loss(pred, target, weights):
    diff = pred - target
    return (diff * diff * weights[None, :]).mean()


def main():
    parser = argparse.ArgumentParser()

    default_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(default_base, "data"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(default_base, "models"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] data_dir: {args.data_dir}")
    print(f"[INFO] out_dir : {args.out_dir}")

    dataset = WorldDataset(args.data_dir)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = WorldMLP(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_weights = make_loss_weights(device)

    print("\n[loss weights]")
    for i, w in enumerate(loss_weights.detach().cpu().numpy()):
        if w != 1.0:
            print(f"dim {i:02d}: weight={w}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_count = 0

        # 参考用：重点次元のMSE
        dim_sse = {
            6: 0.0,
            7: 0.0,
            8: 0.0,
            13: 0.0,
        }

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = weighted_mse_loss(
                pred=pred,
                target=y,
                weights=loss_weights,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

            with torch.no_grad():
                diff = pred - y
                for dim in dim_sse.keys():
                    dim_sse[dim] += float(torch.sum(diff[:, dim] ** 2).item())

        avg_loss = total_loss / max(total_count, 1)

        msg = f"[epoch {epoch:03d}] loss={avg_loss:.8f}"

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            vel_rmse = np.sqrt(dim_sse[6] / max(total_count, 1))
            perp_rmse = np.sqrt(dim_sse[7] / max(total_count, 1))
            head_rmse = np.sqrt(dim_sse[8] / max(total_count, 1))
            vr_rmse = np.sqrt(dim_sse[13] / max(total_count, 1))

            msg += (
                f" | norm_rmse "
                f"vel={vel_rmse:.5f} "
                f"perp={perp_rmse:.5f} "
                f"head={head_rmse:.5f} "
                f"vratio={vr_rmse:.5f}"
            )

        print(msg)

    save_path = os.path.join(args.out_dir, "world_mlp.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "hidden_dim": args.hidden_dim,
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
            "target": "delta_obs_norm",
            "loss": "weighted_mse",
            "loss_weights": loss_weights.detach().cpu().numpy(),
        },
        save_path,
    )

    print(f"\n✅ saved: {save_path}")


if __name__ == "__main__":
    main()