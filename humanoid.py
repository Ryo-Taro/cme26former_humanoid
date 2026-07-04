import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


# =====================
# Reward Logger
# =====================
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_reward += reward

        if done:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        return True


# =====================
# checkpoint
# =====================
def save_checkpoint(model, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"model_{timestamp}")
    model.save(path)
    print(f"✅ checkpoint保存: {path}.zip")


def load_latest_checkpoint(env, save_dir="checkpoints"):
    if not os.path.exists(save_dir):
        return None

    files = [f for f in os.listdir(save_dir) if f.endswith(".zip")]
    if len(files) == 0:
        return None

    latest = sorted(files)[-1]
    path = os.path.join(save_dir, latest)

    print(f"✅ 最新checkpointロード: {path}")
    return PPO.load(path, env=env)


# =====================
# シンプルReward Wrapper
# =====================
class SimpleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # pure RL（without controller）
        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        q = qpos[7:]
        qd = qvel[6:]
        height = qpos[2]

        weight_q = np.array([1.0] * len(q))  # 姿勢維持の重み
        weight_q[1] = 15.0  # 姿勢維持の重み

        # 報酬
        reward = 0.0
        reward -= 0.1 * np.sum(weight_q * np.square(q)) # 姿勢維持
        reward += 0.01 * height               # 高さ維持
        # reward -= 0.00001 * np.sum(qd**2)      # 動きすぎ罰

        # 転倒
        if height < 0.8:
            reward -= 20.0
            terminated = True

        return obs, reward, terminated, truncated, info


# =====================
# 環境
# =====================
env_train = gym.make("Humanoid-v5")
env_train = SimpleRewardWrapper(env_train)


# =====================
# モデル
# =====================
model = load_latest_checkpoint(env_train)

if model is None:
    print("✅ 新規モデル")

    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.05,  # 探索強め
    )
else:
    print("✅ 再開")


# =====================
# 学習
# =====================
callback = RewardLoggerCallback()

model.learn(
    total_timesteps = 500_000,
    callback=callback
)

save_checkpoint(model)
env_train.close()


# =====================
# 学習報酬プロット
# =====================
plt.figure()
plt.plot(callback.episode_rewards)
plt.title("Training Reward (Pure RL)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("training_reward.png")
plt.close()


# =====================
# 評価 + 動画
# =====================
env = gym.make("Humanoid-v5", render_mode="rgb_array")
env = SimpleRewardWrapper(env)

model = load_latest_checkpoint(env)

if model is None:
    model = PPO.load("humanoid_posture_rl")

obs, _ = env.reset()

frames = []
action_history = []
qpos_history = []
qvel_history = []   # ★追加

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    frames.append(env.render())

    data = env.unwrapped.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()   # ★追加

    action_history.append(action)
    qpos_history.append(qpos)
    qvel_history.append(qvel)  # ★追加

    if terminated or truncated:
        print(f"✅ 転倒 at step {step}")
        break

env.close()


# =====================
# データ整形
# =====================
qpos_history = np.array(qpos_history)
qvel_history = np.array(qvel_history)

q_history = qpos_history[:, 7:]
qd_history = qvel_history[:, 6:]   # ★追加

action_history = np.array(action_history)


# =====================
# qvel（速度）
# =====================
plt.figure()
for i in range(min(5, qd_history.shape[1])):
    plt.plot(qd_history[:, i], label=f"qvel_{i}")

plt.title("Joint Velocity")
plt.xlabel("Step")
plt.ylabel("Velocity")
plt.legend()
plt.savefig("qvel.png")
plt.close()


# =====================
# action
# =====================
plt.figure()
for i in range(min(5, action_history.shape[1])):
    plt.plot(action_history[:, i], label=f"action_{i}")

plt.title("Action")
plt.xlabel("Step")
plt.ylabel("Action")
plt.legend()
plt.savefig("action.png")
plt.close()