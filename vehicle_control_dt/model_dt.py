import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import matplotlib.pyplot as plt
import seaborn as sns

#計画と行動のマルチタスクモデル
import torch.nn.functional as F
from typing import Optional, Tuple, List

# パラメータ
TIMESTEP_MAX = 4000


# アテンションマップの可視化
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal  # ✅ PyTorch 2.0以降で必須
        )
        self.attn_weights = attn_weights

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# アテンションマップの可視化 step8attn
def visualize_attention(attn_weights, title="Attention Map", layer=0, head=0):
    # attn_weights: list of [B, n_head, T, T]
    attn = attn_weights[layer][0, head]  # shape: (T, T)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn.cpu().numpy(), cmap="viridis")
    plt.title(f"{title} - Layer {layer}, Head {head}")
    plt.xlabel("Key Token")
    plt.ylabel("Query Token")
    plt.show()

# DTのMLP化検証
class DecisionTransformer_MLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.predict_action = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, *_args):
        # args = (timesteps, states, actions, returns_to_go)
        _, states, *_ = _args
        # states: (B, 1, obs_dim)
        x = states[:, 0, :]  # → (B, obs_dim)
        return self.predict_action(x).unsqueeze(1)  # → (B, 1, act_dim)
    
# DTのMLP化検証 復元step1
class DecisionTransformer_Step1(nn.Module):
    def __init__(self, obs_dim, act_dim, embed_dim=128, n_layer=1, n_head=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = 1  # 明示
        self.embed_dim = embed_dim

        # 状態ベクトルの埋め込み
        self.embed_state = nn.Linear(obs_dim, embed_dim)

        # Transformer エンコーダ層（時系列を扱うが、context_len=1なので単一）
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # 行動予測
        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, act_dim)
        )

    def forward(self, timesteps, states, actions=None, returns_to_go=None):
        """
        timesteps: (B, T)
        states: (B, T, obs_dim)
        ※ context_len = 1 を前提
        """
        B, T, D = states.shape
        assert T == 1, "このモデルは context_len = 1 のみ対応です"

        # (B, T, obs_dim) → (B, T, embed_dim)
        x = self.embed_state(states)

        # Transformer に通す前に (T, B, D) に並び替え
        x = x.permute(1, 0, 2)  # → (T=1, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # → (B, T, D)

        # 行動予測（T=1なので x[:, 0, :] でも良い）
        return self.predict_action(x)  # shape: (B, T=1, act_dim)

# DTのMLP化検証 復元step2
class DecisionTransformer_Step2(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=5, embed_dim=128, n_layer=2, n_head=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.embed_dim = embed_dim

        # 状態の埋め込み
        self.embed_state = nn.Linear(obs_dim, embed_dim)

        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # 行動予測
        self.predict_action = nn.Linear(embed_dim, act_dim)

    def forward(self, timesteps, states, actions=None, returns_to_go=None):
        """
        timesteps: (B, T)
        states: (B, T, obs_dim)
        returns, actions は使いません（今の段階では）
        """
        B, T, _ = states.shape
        assert T == self.context_len, f"context_len={self.context_len} に合わせてください"

        # 状態を埋め込み (B, T, D)
        x = self.embed_state(states)

        # Transformer に渡すために (T, B, D) に並び替え
        x = x.permute(1, 0, 2)

        # Transformer通過 (T, B, D)
        x = self.transformer(x)

        # 再び (B, T, D)
        x = x.permute(1, 0, 2)

        # 最後のトークン（最新状態）を使用して行動予測
        x_last = x[:, -1, :]  # (B, D)
        return self.predict_action(x_last).unsqueeze(1)  # (B, 1, act_dim)

# DTのMLP化検証 復元step3
class DecisionTransformer_Step3(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=5, embed_dim=128, n_layer=2, n_head=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_timestep = nn.Embedding(1024, embed_dim)  # 時刻埋め込み（最大長に注意）

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Linear(embed_dim, act_dim)

    def forward(self, timesteps, states, actions=None, returns_to_go=None):
        """
        timesteps: (B, T)
        states: (B, T, obs_dim)
        """
        B, T, _ = states.shape
        assert T == self.context_len, f"context_len={self.context_len} に合わせてください"

        state_embeddings = self.embed_state(states)                     # (B, T, D)
        time_embeddings = self.embed_timestep(timesteps)               # (B, T, D)
        x = state_embeddings + time_embeddings                         # (B, T, D)

        x = x.permute(1, 0, 2)  # → (T, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # → (B, T, D)

        x_last = x[:, -1, :]   # 最後のトークンのみ使用
        return self.predict_action(x_last).unsqueeze(1)  # (B, 1, act_dim)

# DTのMLP化検証 復元step4
class DecisionTransformer_Step4(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=3, embed_dim=128, n_layer=1, n_head=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, act_dim)
        )

    def forward(self, timesteps, states, actions, returns_to_go=None):
        """
        states: (B, T, obs_dim)
        actions: (B, T, act_dim)
        timesteps: unused
        """
        # 埋め込み
        state_embeddings = self.embed_state(states)       # (B, T, D)
        action_embeddings = self.embed_action(actions)    # (B, T, D)

        # (s1, a1, s2, a2, ..., sT) に変換（sTの後のaTは使わない）
        stacked = []
        for t in range(self.context_len):
            stacked.append(state_embeddings[:, t])        # s_t
            if t < self.context_len - 1:
                stacked.append(action_embeddings[:, t])   # a_t
        x = torch.stack(stacked, dim=1)  # (B, 2T-1, D)

        # Transformer
        x = x.permute(1, 0, 2)  # → (L, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # → (B, L, D)

        # 最後のstate位置（s_T）を抽出して行動を予測
        final_state_index = 2 * self.context_len - 2
        return self.predict_action(x[:, final_state_index].unsqueeze(1))  # (B, 1, act_dim)

# DTのMLP化検証 復元step5
class DecisionTransformer_Step5(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=3, embed_dim=128, n_layer=2, n_head=2):
        super().__init__()
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embed_timestep = nn.Embedding(TIMESTEP_MAX, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Linear(embed_dim, act_dim)

    def forward(self, timesteps, states, actions, returns_to_go):
        B, T, _ = states.shape

        time_emb = self.embed_timestep(timesteps)  # (B, T, D)
        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb
        return_emb = self.embed_return(returns_to_go) + time_emb

        # [rtg_1, state_1, action_1, ..., rtg_T, state_T, action_T]
        stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
        x = stacked.view(B, T * 3, self.embed_dim)  # (B, 3T, D)

        x = x.permute(1, 0, 2)  # (3T, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, 3T, D)

        # 状態位置を抽出：rtg_1, state_1, action_1, ...
        state_tokens = x[:, 1::3]  # (B, T, D)
        return self.predict_action(state_tokens)  # (B, T, act_dim)


# DTのMLP化検証 復元step6
class DecisionTransformer_Step6(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=3, embed_dim=128, n_layer=1, n_head=1):
        super().__init__()
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_timestep = nn.Embedding(4096, embed_dim)  # ステップ数が4096以上なら調整

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, act_dim)
        )

    def forward(self, timesteps, states, actions, returns_to_go):
        B, T, _ = states.shape  # T = context_len

        # timestep embedding
        time_embed = self.embed_timestep(timesteps)  # (B, T, D)

        # individual embeddings
        state_embed = self.embed_state(states) + time_embed
        action_embed = self.embed_action(actions) + time_embed
        return_embed = self.embed_return(returns_to_go) + time_embed

        # stack [r1, s1, a1, ..., rT, sT, aT]
        stacked = torch.stack((return_embed, state_embed, action_embed), dim=2)  # (B, T, 3, D)
        stacked = stacked.reshape(B, T * 3, self.embed_dim)  # (B, 3T, D)

        # Transformer expects (L, B, D)
        x = stacked.permute(1, 0, 2)  # (3T, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, 3T, D)

        # Extract state token positions: [s1, s2, ..., sT]
        state_tokens = x[:, 1::3]  # index 1, 4, 7, ...

        # Predict actions from state token positions
        return self.predict_action(state_tokens)  # (B, T, act_dim)

# DTのMLP化検証 復元step7
class DecisionTransformer_Step7(nn.Module):
#    def __init__(self, obs_dim, act_dim, context_len=3, embed_dim=128, n_layer=2, n_head=2):
    def __init__(self, obs_dim, act_dim, context_len=3, embed_dim=128, n_layer=4, n_head=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.embed_dim = embed_dim

        # 各種埋め込み
        self.embed_timestep = nn.Embedding(TIMESTEP_MAX, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, act_dim)
        )

    def forward(self, timesteps, states, actions, returns_to_go):
        B, T = states.shape[0], states.shape[1]

        # 🔧 修正：timestepsのshapeが(B, T, 1)だった場合に備える
        if timesteps.ndim == 3:
            timesteps = timesteps.squeeze(-1)  # (B, T, 1) → (B, T)

        # --- 埋め込み ---
        time_emb = self.embed_timestep(timesteps)  # (B, T, D)

        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb
        return_emb = self.embed_return(returns_to_go) + time_emb

        # --- トークン順序：r₁, s₁, a₁, r₂, s₂, a₂, ..., rT, sT, aT ---
        stacked = torch.stack((return_emb, state_emb, action_emb), dim=2)  # (B, T, 3, D)
        stacked = stacked.view(B, -1, self.embed_dim)  # (B, 3T, D)

        x = self.dropout(stacked)
        x = x.permute(1, 0, 2)  # → (L=3T, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # → (B, 3T, D)

        # --- stateの位置だけ抽出（r₁,s₁,a₁,r₂,... → s₁,s₂,...） ---
        x = x[:, 1::3]  # shape: (B, T, D)

        return self.predict_action(x)  # → (B, T, act_dim)


# DTのMLP化検証 復元step8
class DecisionTransformer_Step8(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=1, embed_dim=128, n_layer=2, n_head=4):#段階的拡張の開始設定
#    def __init__(self, obs_dim, act_dim, context_len=5, embed_dim=128, n_layer=3, n_head=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embed_timestep = nn.Embedding(TIMESTEP_MAX, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, act_dim)
        )

    def forward(self, timesteps, states, actions, returns_to_go):
        B, T = states.shape[0], states.shape[1]
        if timesteps.ndim == 3:
            timesteps = timesteps.squeeze(-1)  # (B, T, 1) → (B, T)

        # --- 時刻埋め込み ---
        time_emb = self.embed_timestep(timesteps)  # (B, T, D)

        # --- トークンごとの埋め込み + 時刻 ---
        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb
        return_emb = self.embed_return(returns_to_go) + time_emb

        # --- 新トークン順序: state → action → return ---
        # → shape: (B, T, 3, D)
        stacked = torch.stack((state_emb, action_emb, return_emb), dim=2)
        # → shape: (B, 3T, D)
        stacked = stacked.view(B, -1, self.embed_dim)

        x = self.dropout(stacked)
        x = x.permute(1, 0, 2)  # (3T, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, 3T, D)

        # --- state位置（s₀,s₁,…）だけ抽出：0, 3, 6, ...
        x = x[:, 0::3]  # → (B, T, D)

        return self.predict_action(x)  # → (B, T, act_dim)


#最新アテンションマップ対応(step8ベース)
#動作は問題ないがモンキーパッチによる可視化方法が悪く、エラー扱いが邪魔なので封印しておく
#class DecisionTransformer_Step8_NewAttn(nn.Module):
#    def __init__(self, obs_dim, act_dim, context_len=1, embed_dim=128, n_layer=2, n_head=4, timestep_max=1024):
#        super().__init__()
#        self.obs_dim = obs_dim
#        self.act_dim = act_dim
#        self.context_len = context_len
#        self.embed_dim = embed_dim
#        self.n_head = n_head
#
#        self.embed_timestep = nn.Embedding(timestep_max, embed_dim)
#        self.embed_return   = nn.Linear(1, embed_dim)
#        self.embed_state    = nn.Linear(obs_dim, embed_dim)
#        self.embed_action   = nn.Linear(act_dim, embed_dim)
#        self.dropout        = nn.Dropout(0.1)
#
#        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, batch_first=False)
#        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
#
#        self.predict_action = nn.Sequential(nn.Linear(embed_dim, act_dim))
#
#        # ---- attention capture (monkey-patch方式) ----
#        self._attn_weights = []          # list of (B, heads, Lq, Lk)
#        self._attn_patches = []          # list of (attn_module, orig_forward)
#        self._self_attn_layers = [self.transformer.layers[i].self_attn for i in range(n_layer)]
#
#    # ★★ これだけを使う：重みを必ず返すよう forward をラップ
#    def enable_attention_hook(self, enabled: bool = True):
#        # 既存パッチ解除
#        for attn, orig_fwd in self._attn_patches:
#            attn.forward = orig_fwd
#        self._attn_patches.clear()
#        self._attn_weights.clear()
#
#        if not enabled:
#            return
#
#        def make_wrapped(attn_mod):
#            orig_forward = attn_mod.forward
#            def wrapped_forward(query, key, value, **kwargs):
#                kwargs["need_weights"] = True
#                kwargs["average_attn_weights"] = False  # ヘッド別
#                out, w = orig_forward(query, key, value, **kwargs)
#                # 形状統一
#                if w is not None:
#                    if w.dim() == 2:      # (Lq, Lk)
#                        w = w.unsqueeze(0).unsqueeze(0)          # (1,1,Lq,Lk)
#                    elif w.dim() == 3:    # (B, Lq, Lk)
#                        w = w.unsqueeze(1)                        # (B,1,Lq,Lk)
#                    self._attn_weights.append(w.detach().cpu())
#                return out, w
#            return orig_forward, wrapped_forward
#
#        for attn in self._self_attn_layers:
#            orig_fwd, wrapped_fwd = make_wrapped(attn)
#            self._attn_patches.append((attn, orig_fwd))
#            attn.forward = wrapped_fwd
#            setattr(attn, "_patched", True)  # ← デバッグ用フラグ
#
#    def disable_attention_hook(self):
#        for attn, orig_fwd in self._attn_patches:
#            attn.forward = orig_fwd
#        self._attn_patches.clear()
#
#    def get_collected_attn(self):
#        return self._attn_weights
#
#    def forward(self, timesteps, states, actions, returns_to_go):
#        B, T = states.shape[0], states.shape[1]
#        if timesteps.ndim == 3:
#            timesteps = timesteps.squeeze(-1)
#
#        time_emb   = self.embed_timestep(timesteps)
#        state_emb  = self.embed_state(states) + time_emb
#        action_emb = self.embed_action(actions) + time_emb
#        return_emb = self.embed_return(returns_to_go) + time_emb
#
#        stacked = torch.stack((state_emb, action_emb, return_emb), dim=2)  # (B,T,3,D)
#        x = stacked.view(B, -1, self.embed_dim)                            # (B,3T,D)
#
#        x = self.dropout(x)
#        x = x.permute(1, 0, 2)        # (3T,B,D)
#        x = self.transformer(x)       # ← ここで各層の self_attn がラップされ、重みを収集
#        x = x.permute(1, 0, 2)        # (B,3T,D)
#
#        x = x[:, 0::3]                # stateトークンのみ
#        return self.predict_action(x)
#
#    # DecisionTransformer クラスの中に追記（@torch.no_grad は任意）
#    @torch.no_grad()
#    def get_qkv_weights(self):
#        """
#        各層の Q/K/V/O の重みを返す。
#        返り値: list[ { "W_q","b_q","W_k","b_k","W_v","b_v","W_o","b_o" } ] 
#        すべて torch.Tensor（biasは None の事もあり）。
#        """
#        mats = []
#        for layer in self.transformer.layers:
#            attn = layer.self_attn
#
#            # PyTorchの実装差異に配慮（通常は in_proj_weight がある）
#            if getattr(attn, "in_proj_weight", None) is not None:
#                # in_proj_weight = [W_q; W_k; W_v] 連結 (3D, D)
#                W_in = attn.in_proj_weight.detach()
#                b_in = attn.in_proj_bias.detach() if attn.in_proj_bias is not None else None
#                W_q, W_k, W_v = torch.split(W_in, self.embed_dim, dim=0)
#                if b_in is not None:
#                    b_q, b_k, b_v = torch.split(b_in, self.embed_dim, dim=0)
#                else:
#                    b_q = b_k = b_v = None
#            else:
#                # まれに分離プロジェクションのケース（保険）
#                W_q = getattr(attn, "q_proj_weight").detach()
#                W_k = getattr(attn, "k_proj_weight").detach()
#                W_v = getattr(attn, "v_proj_weight").detach()
#                b_q = getattr(attn, "q_proj_bias", None)
#                b_k = getattr(attn, "k_proj_bias", None)
#                b_v = getattr(attn, "v_proj_bias", None)
#                if b_q is not None: b_q = b_q.detach()
#                if b_k is not None: b_k = b_k.detach()
#                if b_v is not None: b_v = b_v.detach()
#
#            W_o = attn.out_proj.weight.detach()
#            b_o = attn.out_proj.bias.detach() if attn.out_proj.bias is not None else None
#
#            mats.append({
#                "W_q": W_q, "b_q": b_q,
#                "W_k": W_k, "b_k": b_k,
#                "W_v": W_v, "b_v": b_v,
#                "W_o": W_o, "b_o": b_o
#            })
#        return mats
#
#    @torch.no_grad()
#    def get_state_embed_weight(self):
#        """観測ベクトル → 埋め込み への線形層の重み (D, obs_dim) を返す"""
#        return self.embed_state.weight.detach()




#計画と行動のマルチタスクモデル
#class DecisionTransformer_Step9(nn.Module):
class DecisionTransformer(nn.Module):

    #※ timestep_vocab は TIMESTEP_MAX 以上の2の乗数で
    def __init__(self, obs_dim, act_dim, context_len=1, embed_dim=128, n_layer=2, n_head=4, timestep_vocab=4096, plan_M: int = 3, use_focus: bool = False, force_clip: float = 0.8, idle_throttle_init: float = 0.0908,wp_dim: int = 5):
#   def __init__(self, obs_dim, act_dim, context_len=1, embed_dim=128, n_layer=2, n_head=4, timestep_max=1024):
  
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.embed_dim = embed_dim
        self.n_head = n_head

        #計画と行動のマルチタスクモデル
        self.plan_M = plan_M
        self.use_focus = use_focus
        self.wp_dim = wp_dim


        #計画と行動のマルチタスクモデル
        self.embed_timestep = nn.Embedding(timestep_vocab, embed_dim)
#        self.embed_timestep = nn.Embedding(timestep_max, embed_dim)
        
        self.embed_return   = nn.Linear(1, embed_dim)
        self.embed_state    = nn.Linear(obs_dim, embed_dim)
        self.embed_action   = nn.Linear(act_dim, embed_dim)

        #計画と行動のマルチタスクモデル WPプレフィクス用の埋め込み（dx,dy,s,κ,width）
        self.embed_wp = nn.Linear(wp_dim, embed_dim)

        self.dropout        = nn.Dropout(0.1)

        #ボトルネック認識とVmax魂の注入 1.1型埋め込み（state/action/rtg を識別して注意の分業を誘導)
        self.embed_type = nn.Embedding(num_embeddings=4, embedding_dim=embed_dim)
        self.type_state_with_vmax = 1
        self.type_action = 2
        self.type_rtg    = 3

        self.force_clip = force_clip
        self.idle_throttle_init = idle_throttle_init
        self._probe_force = True

#計画と行動のマルチタスクモデル
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, batch_first=True)
#       encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, batch_first=False)
 

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)


#計画と行動のマルチタスクモデル
        self.predict_action = nn.Linear(embed_dim, act_dim)
        # 参照ライン（計画）ヘッド：WPプレフィクスのプール表現から2M次元を出力
        self.predict_plan = nn.Linear(embed_dim, 2 * plan_M)
        # （任意）フォーカスヘッド：各WPトークンに重み
        if use_focus:
            self.focus_head = nn.Linear(embed_dim, 1)
        else:
            self.focus_head = None
#       self.predict_action = nn.Sequential(nn.Linear(embed_dim, act_dim))


		#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる 最後に：fresh 初期化として throttle 側バイアスを設定 ---
        # idle を [0, force_clip] → p∈(0,1) に正規化して logit
        p = max(1e-6, min(1 - 1e-6, self.idle_throttle_init / self.force_clip))
        b = math.log(p / (1 - p))
        with torch.no_grad():
            if self.act_dim >= 2:  # [steer, throttle] 前提
                self.predict_action.bias[0].zero_()  # steer 側は 0 初期化
                self.predict_action.bias[1].fill_(b) # throttle 側を logit(idle/F) に

#計画と行動のマルチタスクモデル
        # ---- attention capture (hook方式) ----
        self._attn_weights: List[torch.Tensor] = []     # list of (B, heads, L, L)
        self._attn_handles: List[torch.utils.hooks.RemovableHandle] = []
#        # ---- attention capture (monkey-patch方式) ----
#        self._attn_weights = []          # list of (B, heads, Lq, Lk)
#        self._attn_patches = []          # list of (attn_module, orig_forward)
#        self._self_attn_layers = [self.transformer.layers[i].self_attn for i in range(n_layer)]

    #計画と行動のマルチタスクモデル
    @staticmethod
    def _build_prefix_causal_mask(K: int, T3: int, device) -> torch.Tensor:
        """
        K: WPトークン数, T3: 時系列トークン総数（state,action,rtgの3T）
        マスク仕様：
          - WP（先頭K）は全てにフルアクセス（双方向）
          - 時系列は因果（未来を見ない）
          - 時系列→WP は見える（WPはプロンプト）
        """
        L = K + T3
        mask = torch.zeros((L, L), device=device)  # 0=許可, -inf=禁止
        if T3 > 0:
            tri = torch.triu(torch.full((T3, T3), float('-inf'), device=device), diagonal=1)
            mask[K:, K:] = tri  # 時系列は因果
            # 時系列→WP は許可（既に0）
        # WP行（先頭K）は全許可（既に0）
        return mask


    # ★★ これだけを使う：重みを必ず返すよう forward をラップ
    def enable_attention_hook(self, enabled: bool = True):

#計画と行動のマルチタスクモデル
        # 既存hook解除
        self.disable_attention_hook()
        self._attn_weights.clear()
        if not enabled:
            return
        # 各EncoderLayerの MultiheadAttention に pre/post hook を登録
        def pre_hook(module, args, kwargs):
            kw = dict(kwargs) if kwargs is not None else {}
            kw["need_weights"] = True
            kw["average_attn_weights"] = False
            return args, kw
        def post_hook(module, inputs, outputs):
            # outputs=(attn_output, attn_weights)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                w = outputs[1]
                if w is not None:
                    # 形状統一: (B, heads, L, L)
                    if w.dim() == 2:
                        w = w.unsqueeze(0).unsqueeze(0)
                    elif w.dim() == 3:
                        w = w.unsqueeze(1)
                    self._attn_weights.append(w.detach().cpu())
        for layer in getattr(self.transformer, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, nn.MultiheadAttention):
                self._attn_handles.append(attn.register_forward_pre_hook(pre_hook))
                self._attn_handles.append(attn.register_forward_hook(post_hook))
#        # 既存パッチ解除
#        for attn, orig_fwd in self._attn_patches:
#            attn.forward = orig_fwd
#        self._attn_patches.clear()
#        self._attn_weights.clear()
#
#        if not enabled:
#            return
#        def make_wrapped(attn_mod):
#            orig_forward = attn_mod.forward
#            def wrapped_forward(query, key, value, **kwargs):
#                kwargs["need_weights"] = True
#                kwargs["average_attn_weights"] = False  # ヘッド別
#                out, w = orig_forward(query, key, value, **kwargs)
#                # 形状統一
#                if w is not None:
#                    if w.dim() == 2:      # (Lq, Lk)
#                        w = w.unsqueeze(0).unsqueeze(0)          # (1,1,Lq,Lk)
#                    elif w.dim() == 3:    # (B, Lq, Lk)
#                        w = w.unsqueeze(1)                        # (B,1,Lq,Lk)
#                    self._attn_weights.append(w.detach().cpu())
#                return out, w
#            return orig_forward, wrapped_forward
#
#        for attn in self._self_attn_layers:
#            orig_fwd, wrapped_fwd = make_wrapped(attn)
#            self._attn_patches.append((attn, orig_fwd))
#            attn.forward = wrapped_fwd
#            setattr(attn, "_patched", True)  # ← デバッグ用フラグ

    def disable_attention_hook(self):
#計画と行動のマルチタスクモデル
        for h in self._attn_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._attn_handles.clear()
#        for attn, orig_fwd in self._attn_patches:
#            attn.forward = orig_fwd
#        self._attn_patches.clear()

    def get_collected_attn(self):
        return self._attn_weights

#計画と行動のマルチタスクモデル
    def forward(self, timesteps, states, actions, returns,
                    wp: Optional[torch.Tensor] = None,
                    return_plan: bool = True,
                    return_focus: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
#   def forward(self, timesteps, states, actions, returns_to_go):

#計画と行動のマルチタスクモデル
        B, T, _ = states.shape
#       B, T = states.shape[0], states.shape[1]

#ボトルネック認識とVmax魂の注入 1.2 forward：型埋め込みを加算＋安全チェック
        # 形の安全チェック（obs拡張後は 19 次元が期待）
        if states.shape[-1] != self.obs_dim:
            raise RuntimeError(f"[DecisionTransformer] obs_dim mismatch: got {states.shape[-1]} but model expects {self.obs_dim}")

        if timesteps.ndim == 3:
            timesteps = timesteps.squeeze(-1)

        time_emb   = self.embed_timestep(timesteps)

#計画と行動のマルチタスクモデル

#ボトルネック認識とVmax魂の注入 1.2 forward：型埋め込みを加算＋安全チェック
        s_tok = self.embed_state(states) + time_emb + self.embed_type.weight[self.type_state_with_vmax]
        a_tok = self.embed_action(actions) + time_emb + self.embed_type.weight[self.type_action]
        r_tok = self.embed_return(returns) + time_emb + self.embed_type.weight[self.type_rtg]
#        s_tok = self.embed_state(states) + time_emb
#        a_tok = self.embed_action(actions) + time_emb
#        r_tok = self.embed_return(returns) + time_emb


#計画と行動のマルチタスクモデル
        x_time = torch.stack([s_tok, a_tok, r_tok], dim=2).reshape(B, 3*T, self.embed_dim)
        # WPプレフィクス（無ければ長さ0）
        if wp is not None and wp.numel() > 0:
            x_wp = self.embed_wp(wp)  # (B, K, D)
            K = x_wp.shape[1]
            x = torch.cat([x_wp, x_time], dim=1)  # (B, K+3T, D)
        else:
            x = x_time
            K = 0
        # hook使用時：前回の記録をクリア
        if self._attn_handles:
            self._attn_weights.clear()
        attn_mask = self._build_prefix_causal_mask(K, x.shape[1]-K, x.device)

        h = self.transformer(x, mask=attn_mask)  # (B, L, D)

        # アクションは各tの"actionトークン位置"に対応
        # 位置: [wp:0..K-1] + [state,action,rtg]xT ⇒ action位置は K + (1 + 3*t)
        action_positions = K + (1 + 3*torch.arange(T, device=states.device))
        h_act = h[:, action_positions, :]  # (B, T, D)

#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる 非負制約 スケール調整
        raw = self.predict_action(h_act)            # (B, T, A=2) = [steer_raw, throttle_raw]

        if getattr(self, "_probe_force", False):
            self._probe_force = False
            print("[FWD] force_clip attr =", getattr(self, "force_clip", None))
            print("[FWD] using class =", self.__class__.__module__, self.__class__.__name__)
            # ロジットから手計算（この forward のコード経由ではない）
            test = torch.sigmoid(torch.tensor([[-2.0554783]], device=h_act.device)) * float(self.force_clip)
            print("[FWD] test throttle (σ(-2.055)*force_clip) =", float(test))

        steer = raw[..., 0:1]                       # ステアは線形のまま（必要なら後述の tanh も可）
        throt = torch.sigmoid(raw[..., 1:2]) * self.force_clip  # [0, FORCE_CLIP]
#アクセルがマイナスになる原因        
        pred_actions = torch.cat([steer, throt], dim=-1)
#        pred_actions = raw#torch.cat([steer, throt], dim=-1)
##        pred_actions = self.predict_action(h_act)  # (B, T, A)

        pred_plan = None
        alpha = None
        if return_plan:
            # WPプレフィクスを平均プーリング（K=0なら時系列先頭を代用）
            if K > 0:
                h_wp = h[:, :K, :]  # (B, K, D)
                pooled = h_wp.mean(dim=1)  # (B, D)
                pred_plan = self.predict_plan(pooled)  # (B, 2M)
                if return_focus and self.focus_head is not None:
                    alpha_logits = self.focus_head(h_wp).squeeze(-1)  # (B, K)
                    alpha = F.softmax(alpha_logits, dim=-1)
            else:
                pooled = h[:, 0, :]
                pred_plan = self.predict_plan(pooled)
        return pred_actions, pred_plan, alpha
#       stacked = torch.stack((state_emb, action_emb, return_emb), dim=2)  # (B,T,3,D)
#       x = stacked.view(B, -1, self.embed_dim)                            # (B,3T,D)
#       x = self.dropout(x)
#       x = x.permute(1, 0, 2)        # (3T,B,D)
#       x = self.transformer(x)       # ← ここで各層の self_attn がラップされ、重みを収集
#       x = x.permute(1, 0, 2)        # (B,3T,D)
#
#       x = x[:, 0::3]                # stateトークンのみ
#       return self.predict_action(x)

    # DecisionTransformer クラスの中に追記（@torch.no_grad は任意）
    @torch.no_grad()
    def get_qkv_weights(self):
        """
        各層の Q/K/V/O の重みを返す。
        返り値: list[ { "W_q","b_q","W_k","b_k","W_v","b_v","W_o","b_o" } ] 
        すべて torch.Tensor（biasは None の事もあり）。
        """
        mats = []
#計画と行動のマルチタスクモデル
        for layer in self.transformer.layers:
            attn = layer.self_attn
            if not isinstance(attn, nn.MultiheadAttention):
                continue
#        for layer in self.transformer.layers:
#            attn = layer.self_attn

            # PyTorchの実装差異に配慮（通常は in_proj_weight がある）
            if getattr(attn, "in_proj_weight", None) is not None:
                # in_proj_weight = [W_q; W_k; W_v] 連結 (3D, D)
                W_in = attn.in_proj_weight.detach()
                b_in = attn.in_proj_bias.detach() if attn.in_proj_bias is not None else None
                W_q, W_k, W_v = torch.split(W_in, self.embed_dim, dim=0)
                if b_in is not None:
                    b_q, b_k, b_v = torch.split(b_in, self.embed_dim, dim=0)
                else:
                    b_q = b_k = b_v = None
            else:
                # まれに分離プロジェクションのケース（保険）
#計画と行動のマルチタスクモデル
                W_q = getattr(attn, "q_proj_weight").detach()
                W_k = getattr(attn, "k_proj_weight").detach()
                W_v = getattr(attn, "v_proj_weight").detach()
                b_q = getattr(attn, "q_proj_bias", None)
                b_k = getattr(attn, "k_proj_bias", None)
                b_v = getattr(attn, "v_proj_bias", None)
#                W_q = getattr(attn, "q_proj_weight").detach()
#                W_k = getattr(attn, "k_proj_weight").detach()
#                W_v = getattr(attn, "v_proj_weight").detach()
#                b_q = getattr(attn, "q_proj_bias", None)
#                b_k = getattr(attn, "k_proj_bias", None)
#                b_v = getattr(attn, "v_proj_bias", None)
                if b_q is not None: b_q = b_q.detach()
                if b_k is not None: b_k = b_k.detach()
                if b_v is not None: b_v = b_v.detach()

            W_o = attn.out_proj.weight.detach()
            b_o = attn.out_proj.bias.detach() if attn.out_proj.bias is not None else None

            mats.append({
                "W_q": W_q, "b_q": b_q,
                "W_k": W_k, "b_k": b_k,
                "W_v": W_v, "b_v": b_v,
                "W_o": W_o, "b_o": b_o
            })
        return mats

    @torch.no_grad()
    def get_state_embed_weight(self):
        """観測ベクトル → 埋め込み への線形層の重み (D, obs_dim) を返す"""
        return self.embed_state.weight.detach()
