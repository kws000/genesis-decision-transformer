import os
import torch
import torch.nn as nn
#import intel_npu_acceleration_library  # これでNPUバックエンドが登録される

from torch.utils.data import DataLoader
import argparse
import pickle
import pickle, numpy as np

from model_dt import DecisionTransformer
from convert_to_dt_format import TIMESTEP_MAX
from train_dt import SequenceDataset  # または TrajectoryDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"

#計画と行動のマルチタスクモデル
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50 #※いまだけ削減
K_WP = 40
PLAN_M = 3
W_ACT = 1.0
W_PLAN = 0.5
W_SMOOTH = 0.01
USE_FOCUS = False
#EPOCHS = 30#50#100#20#100


#最新モデルでリプレイする　別手法
#TRY_CHECKPOINT_PATH = "checkpoints/temp_model.pt"
#TRY_PKL_PATH = "data_dt/trajectories_dt.pkl"
#TRY_NORM_PATH = "data_dt/mean_std.pkl"



def get_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None, -1
    steps = []
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith("step") and f.endswith(".pt"):
            try:
                step_num = int(f[4:-3])
                steps.append((step_num, os.path.join(CHECKPOINT_DIR, f)))
            except:
                continue
    if not steps:
        return None, -1
    steps.sort()
    return steps[-1][1], steps[-1][0]


#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる
def weight_from_returns(returns, floor=0.2):
    r = returns.detach()
    rmin = r.amin(dim=1, keepdim=True)
    rmax = r.amax(dim=1, keepdim=True)
    w = (r - rmin) / (rmax - rmin + 1e-6)     # [0,1]
    return floor + (1.0 - floor) * w          # [floor,1]
#
##計画と行動のマルチタスクモデル
#def _zscore_nonneg(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#    mu = x.mean()
#    sd = x.std(unbiased=False) + eps
#    z = (x - mu) / sd
#    return torch.clamp(z, min=0.0)

#前に進まなくなった直接の原因	非正規化計算のためにmeanとstdをロード
def load_norm_pkl(norm_path):
    with open(norm_path, "rb") as f:
        s = pickle.load(f)
    obs_mean = np.asarray(s["obs_mean"], np.float32)
    obs_std  = np.asarray(s["obs_std"],  np.float32)
    return obs_mean, obs_std


#計画と行動のマルチタスクモデル
def train_external(context_len, n_layer, n_head,norm_path,pkl_path,checkpoint_path,embed_dim=128):
#def train_external(context_len, n_layer, n_head,norm_path,pkl_path,checkpoint_path):

    # データ読み込み
    dataset = SequenceDataset(pkl_path, context_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    obs_dim = dataset[0][0].shape[-1]
    act_dim = dataset[0][1].shape[-1]

    #ボトルネック認識とVmax魂の注入 6.1 
    assert obs_dim == 19, f"obs_dim must be 19 (OBS_V2). got {obs_dim}"

    #前に進まなくなった直接の原因	非正規化計算のためにmeanとstdをロード
    obs_mean_np, obs_std_np = load_norm_pkl(norm_path)  # or load_norm_pkl
    obs_mean_t = torch.tensor(obs_mean_np, device=DEVICE).view(1,1,-1)
    obs_std_t  = torch.tensor(obs_std_np,  device=DEVICE).view(1,1,-1)


    # モデル定義

#計画と行動のマルチタスクモデル
    model = DecisionTransformer(obs_dim, act_dim,
                                context_len=context_len,
                                embed_dim=embed_dim,
                                n_layer=n_layer,
                                n_head=n_head,
                                plan_M=PLAN_M,
                                force_clip=0.8,
                                idle_throttle_init=0.0908,
                                use_focus=USE_FOCUS).to(DEVICE)
#   model = DecisionTransformer(
#       obs_dim=obs_dim,
#       act_dim=act_dim,
#       context_len=context_len,
#       embed_dim=128,
#       n_layer=n_layer,
#       n_head=n_head,
#   ).to(DEVICE)

	#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる スケールが小さすぎる問題
    def _probe_zero_once(tag: str):
        model.eval()
        with torch.no_grad():
            B, T = 1, 1
            t = torch.zeros(B, T, dtype=torch.long, device=DEVICE)
            s = torch.zeros(B, T, model.obs_dim, device=DEVICE)
            a = torch.zeros(B, T, model.act_dim, device=DEVICE)
            r = torch.zeros(B, T, 1,             device=DEVICE)
            pa, _, _ = model(t, s, a, r, wp=None, return_plan=False)
            print(f"[PROBE {tag}] zero-input pred =", pa[0, 0].detach().cpu().numpy())  # steer, throttle
            print(f"[PROBE {tag}] head.bias      =", model.predict_action.bias.detach().cpu().numpy())

	#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる スケールが小さすぎる問題
    _probe_zero_once("before_ckpt")



    # 前回モデルのロード試行
    prev_model_path, prev_step = get_latest_checkpoint()
    if prev_model_path:
        try:
            print(f"🔄 前回モデルをロード: {prev_model_path}")

            state_dict = torch.load(prev_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)

        	#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる スケールが小さすぎる問題
            _probe_zero_once("after_ckpt")

            print("✅ 前回モデルを引き継ぎました")
        except Exception as e:
            print(f"⚠️ 構造が異なるため、前のモデルは使用しません（{e}）")
            print("⚠️ 学習は新規開始されます。")
    else:
        print("⚠️ 前回ステップのモデルが存在しないので新規学習")


#計画と行動のマルチタスクモデル
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
#   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#   loss_fn = nn.MSELoss()



    print("🚀 Training Start")

#計画と行動のマルチタスクモデル
    for epoch in range(EPOCHS):
        total_loss = 0
        for states, actions, returns, timesteps, wp, plan in dataloader:
            timesteps = timesteps.to(DEVICE).long()
            states    = states.to(DEVICE).float()
            actions   = actions.to(DEVICE).float()
            returns   = returns.to(DEVICE).float()

#ボトルネック認識とVmax魂の注入 6.1 WPなしガード（K=0 のケース） 
            wp_in = None
            if wp is not None:
                # 期待形状: (B, K, wp_dim)。K==0 のバッチもあり得る
                if hasattr(wp, "numel") and wp.numel() > 0:
                    wp_in = wp.to(DEVICE).float()

            pred_actions, pred_plan, alpha = model(
                timesteps, states, actions, returns,
                wp=wp_in, return_plan=True, return_focus=USE_FOCUS
            )
#            wp        = wp.to(DEVICE).float()
#            pred_actions, pred_plan, alpha = model(timesteps, states, actions, returns,
#                                                    wp=wp, return_plan=True, return_focus=USE_FOCUS)
            # 行動損失（RTG重み付きBC）

#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる zscore廃止
            w = weight_from_returns(returns, floor=0.2).expand_as(actions)
            #教師アクションと予測アクションの差が行動損失
            L_act = (w * (pred_actions - actions) ** 2).mean()
#            w = _zscore_nonneg(returns)  # (B,T,1)
#            w = w.expand_as(actions)     # (B,T,2)
#            L_act = (w * (pred_actions - actions) ** 2).mean()

            # 計画損失　
            #※planがないことを想定する必要はない            

            # ＊いまだけコミット死刑
            if (pred_plan is None) or (plan is None):
                L_plan = torch.zeros((), device=DEVICE)
            else:            
                plan = plan.to(DEVICE).float()  # (B,2M)
                #教師計画と予測計画の差が計画損失
                L_plan = mse(pred_plan, plan)   

            # 滑らかさ損失（Δa）
            if pred_actions.shape[1] > 1:
                #予測アクセルと前回予測アクセルの差が滑らか損失
                da = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
                L_smooth = (da ** 2).mean()
            else:
                L_smooth = torch.zeros((), device=DEVICE)


#前に進まなくなった直接の原因	非正規化
            # states: (B,T,obs_dim) = obs_norm
            states_phys = states * obs_std_t + obs_mean_t

            vel_phys  = states_phys[..., 6]
            vlim_phys = states_phys[..., 18]

            vlim_phys = torch.clamp(vlim_phys, min=0.05)
            delta = torch.clamp(vel_phys - vlim_phys, min=0.0)

            eps = 0.05
            mask = (delta > eps).float()

            pred_th = pred_actions[..., 1]          # [0, force_clip]

            denom = mask.sum() + 1e-6
            L_sm = ((pred_th - 0.0)**2 * mask).sum() / denom

##ボトルネック認識とVmax魂の注入 6.2 安全余裕損失 L_sm（速度上限超過時のみアクセルを監督）
#            # OBS_V2: vel=idx6, limit_v_target=idx18
#            vel  = states[..., 6]
#            vlim = states[..., 18]
#            delta = torch.clamp(vel - vlim, min=0.0)           # 超過量
#            accel_idx = 1  # action=[steer, accel]
#            # 減速度の目安（単純比例でOK）：過剰に強くしない
#            a_des = (-1.0 * delta / (vlim.abs() + 1e-3)).clamp(-1.0, 1.0)
#            mask = (delta > 0).float()
#            denom = mask.sum() + 1e-6
#            L_sm  = ((pred_actions[..., accel_idx] - a_des)**2 * mask).sum() / denom

#前に進まなくなった直接の原因 L_sm の大きすぎが原因で減速している
            loss = W_ACT * L_act + W_PLAN * L_plan + W_SMOOTH * L_smooth + 0.0 * L_sm
#            loss = W_ACT * L_act + W_PLAN * L_plan + W_SMOOTH * L_smooth + 0.2 * L_sm
#            loss = W_ACT * L_act + W_PLAN * L_plan + W_SMOOTH * L_smooth

            optimizer.zero_grad()
            # 全損失から誤差逆伝搬を行い、勾配を調整する
            loss.backward()

            #ボトルネック認識とVmax魂の注入 6.2 安全余裕損失 L_sm（速度上限超過時のみアクセルを監督）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


            optimizer.step()
            total_loss += float(loss.item())
            avg_loss = total_loss / max(1, len(dataloader))

#前に進まなくなった直接の原因	非正規化
            viol_rate = (vel_phys > vlim_phys).float().mean().item()
#            viol_rate = (vel > vlim).float().mean().item()
            print(
                f"Epoch {epoch+1:03d} | L={avg_loss:.5f} "
                f"| L_act={L_act.item():.4f} | L_plan={L_plan.item():.4f} "
                f"| L_smooth={L_smooth.item():.4f} | L_sm={L_sm.item():.4f} "
                f"| viol={viol_rate:.3f}"
            )

    # 保存
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    step_id = prev_step + 1

    # ここで暫定モデル temp_model.pt が保存される

#最新モデルでリプレイする　別手法
    save_path = checkpoint_path
#    save_path = os.path.join(CHECKPOINT_DIR, f"temp_model.pt")

    torch.save(model.state_dict(), save_path)
    print(f"✅ 暫定モデルを保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_len", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--n_head", type=int, required=True)
    #最新モデルでリプレイする　別手法
    parser.add_argument("--norm_path", type=str, required=True)
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    args = parser.parse_args()

    train_external(args.context_len, args.n_layer, args.n_head,
                   args.norm_path,args.pkl_path,args.checkpoint_path)
