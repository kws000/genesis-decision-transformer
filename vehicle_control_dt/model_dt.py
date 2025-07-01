import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

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
#class DecisionTransformer_Step8(nn.Module):
class DecisionTransformer(nn.Module):
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

# DTのMLP化検証 復元step8attn
class DecisionTransformer_Step8_Attn(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=5, embed_dim=128, n_layer=3, n_head=4):
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

        # カスタム Transformer Encoder を使用
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
            for _ in range(n_layer)
        ])

        self.predict_action = nn.Linear(embed_dim, act_dim)

    def forward(self, timesteps, states, actions, returns_to_go):
        B, T = states.shape[0], states.shape[1]
        if timesteps.ndim == 3:
            timesteps = timesteps.squeeze(-1)

        time_emb = self.embed_timestep(timesteps)
        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb
        return_emb = self.embed_return(returns_to_go) + time_emb

        stacked = torch.stack((state_emb, action_emb, return_emb), dim=2)
        stacked = stacked.view(B, -1, self.embed_dim)

        x = self.dropout(stacked)
        x = x.permute(1, 0, 2)  # (seq_len, B, D)

        for layer in self.layers:
            x = layer(x)

        x = x.permute(1, 0, 2)  # (B, seq_len, D)
        x = x[:, 0::3]  # 予測に使うstateトークン位置
        return self.predict_action(x)

    def get_attention_maps(self):
        # 各レイヤーから注意重みを取得
        return [layer.attn_weights for layer in self.layers]


# DTのMLP化検証 復元
class DecisionTransformer_org(nn.Module):
    def __init__(self, obs_dim, act_dim, context_len=20, embed_dim=128, n_layer=4, n_head=4):
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
        """
        timesteps: (B, T)
        states: (B, T, obs_dim)
        actions: (B, T, act_dim)
        returns_to_go: (B, T, 1)
        """
        B, T = states.shape[0], states.shape[1]

        # 🔧 修正：timestepsのshapeが(B, T, 1)だった場合に備える
        if timesteps.ndim == 3:
            timesteps = timesteps.squeeze(-1)  # (B, T, 1) → (B, T)

        # Embeddings
        time_embeddings = self.embed_timestep(timesteps)                # (B, T, D)
        state_embeddings = self.embed_state(states) + time_embeddings  # (B, T, D)
        action_embeddings = self.embed_action(actions) + time_embeddings
        return_embeddings = self.embed_return(returns_to_go) + time_embeddings

        # Stack as (r1, s1, a1, r2, s2, a2, ..., rT, sT, aT)
        stacked_inputs = torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=2)
        stacked_inputs = stacked_inputs.view(B, -1, self.embed_dim)  # (B, 3*T, D)

        x = self.dropout(stacked_inputs)
        x = x.permute(1, 0, 2)  # (L, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, L, D)

        # extract state token positions → predict next action
        x = x[:, 1::3]  # s1, s2, ..., sT

        return self.predict_action(x)  # (B, T, act_dim)
