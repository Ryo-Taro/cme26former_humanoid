import os
import re
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import SAC

# =====================================
# あなたの RewardWrapper をそのままコピー
# =====================================
class SimpleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        qd = qvel[6:]
        height = qpos[2]
        com = data.subtree_com[0]
        torso = qpos[3:7]

        reward = 0.0

        reward -= 20.0 * (com[0]**2 + com[1]**2)
        reward -= 5.0 * np.sum(torso[1:]**2)
        reward += 0.5 * height
        reward -= 1e-5 * np.sum(qd**2)
        reward += 1.0

        if height < 0.8:
            reward -= 20.0
            terminated = True

        return obs, reward, terminated, truncated, info


# =====================================
# 設定
# =====================================
checkpoint_dir = "Logs/20260706_005225/checkpoints"   # ←変更
n_eval_episodes = 30

env = gym.make("Humanoid-v5")
env = SimpleRewardWrapper(env)

steps = []
mean_rewards = []
std_rewards = []

# model_2000000.zip のようなファイルだけ取得
files = sorted(
    [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

for file in files:

    model_path = os.path.join(checkpoint_dir, file)
    model = SAC.load(model_path)

    rewards = []

    for ep in range(n_eval_episodes):

        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    step = int(re.findall(r"\d+", file)[0])

    steps.append(step)
    mean_rewards.append(np.mean(rewards))
    std_rewards.append(np.std(rewards))

    print(f"{step:>10} : {np.mean(rewards):8.2f}")

env.close()

# =====================================
# グラフ
# =====================================
plt.figure(figsize=(8,5))

plt.plot(steps, mean_rewards, marker="o")

plt.fill_between(
    steps,
    np.array(mean_rewards)-np.array(std_rewards),
    np.array(mean_rewards)+np.array(std_rewards),
    alpha=0.3,
)

plt.xlabel("Training Timesteps")
plt.ylabel("Average Reward")
plt.title("Checkpoint Evaluation")
plt.grid(True)

plt.savefig("checkpoint_reward_curve.png", dpi=300)
plt.show()