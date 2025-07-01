from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from genesis_gym_env import GenesisEnv

from stable_baselines3.common.callbacks import CheckpointCallback



checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path='./checkpoints/',
    name_prefix='sac_genesis'
)


# 1. 環境を作成
env = GenesisEnv()

# 不要とのこと
# # 2. Gym環境が正しく実装されているか確認（初回のみ）
# check_env(env)


load_start = False#True#False

if load_start:
    # 2.ロードしてから始める場合
    model = SAC.load("checkpoints/sac_genesis_20000_steps", env=env)
else:
    # 3. SACエージェントを作成
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./sac_tensorboard/"
    )

# 4. 学習開始d
model.learn(total_timesteps=100_000, callback=checkpoint_callback)

# 5. モデル保存
model.save("sac_genesis")

# 6. 環境終了
env.close()
