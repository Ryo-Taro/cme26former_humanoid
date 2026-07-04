import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import Wrapper


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

    latest_file = sorted(files)[-1]
    latest_path = os.path.join(save_dir, latest_file)

    print(f"✅ 最新checkpointロード: {latest_path}")
    return PPO.load(latest_path, env=env)


# =====================
# Target Pose
# =====================
class TargetPose:
    def __init__(self, env):
        data = env.unwrapped.data
        self.qpos_target = data.qpos.copy()
        self.com = np.array([0.0, 0.0])


# =====================
# Controller
# =====================
class PostureCOMController:
    def __init__(self, target):
        self.target = target

    def get_action(self, env):
        data = env.unwrapped.data

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        q = qpos[7:]
        qd = qvel[6:]
        q_target = self.target.qpos_target[7:]

        action = np.zeros_like(q)

        # ↓ 弱めたPD
        action += -20.0 * (q - q_target) - 2.0 * qd

        # COM制御（弱め）
        com = data.subtree_com[0]
        com_vel = data.cvel[0][:2]

        ex = com[0]
        ey = com[1]

        action[0] += -5 * ex - 2 * com_vel[0]
        action[1] += -10 * ey - 2 * com_vel[1]

        return action


# =====================
# Wrapper
# =====================
class PostureResidualWrapper(Wrapper):
    def __init__(self, env, target):
        super().__init__(env)
        self.target = target
        self.controller = PostureCOMController(target)

    def step(self, action_rl):
        base_action = self.controller.get_action(self.env)

        # ✅ residualを小さく
        action = base_action + 0.1 * action_rl
        action = np.clip(action, -1.0, 1.0)

        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        q = qpos[7:]
        qd = qvel[6:]
        q_target = self.target.qpos_target[7:]

        q_error = q - q_target
        com = data.subtree_com[0]
        height = qpos[2]

        weights_error = np.ones_like(q_error)
        weights_error[0] = 8.0
        weights_error[2] = 6.0
        weights_error[16] = 8.0
        weights_error[1] = 5.0

        # 報酬
        reward = 0.0
        reward -= 0.5 * np.sum(weights_error * (q_error**2))
        # reward -= 0.01 * np.sum(qd**2)
        reward -= 0.5 * np.sum((com[:2])**2)
        reward -= 0.001 * np.sum(action_rl**2)
        reward += 0.5   # 生存報酬

        if height < 0.8:
            reward -= 5.0
            terminated = True

        return obs, reward, terminated, truncated, info


# =====================
# 環境
# =====================
env_tmp = gym.make("Humanoid-v5")
target = TargetPose(env_tmp)
env_tmp.close()

env_train = gym.make("Humanoid-v5")
env_train = PostureResidualWrapper(env_train, target)


# =====================
# モデル
# =====================
model = load_latest_checkpoint(env_train)

if model is None:
    print("✅ 新規モデル作成")

    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.98,           # ✅ 長期重視
        gae_lambda=0.95,
        clip_range=0.2,       # ✅ 安定化
        ent_coef=0.05,       # ✅ 探索
    )
else:
    print("✅ checkpointから再開")


# =====================
# 学習
# =====================
callback = RewardLoggerCallback()

model.learn(
    total_timesteps=180_000,
    callback=callback
)

save_checkpoint(model)
env_train.close()


# =====================
# 学習報酬プロット
# =====================
plt.figure()
plt.plot(callback.episode_rewards)
plt.title("Training Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("training_reward.png")
plt.close()


# =====================
# 評価 + 動画
# =====================
env = gym.make("Humanoid-v5", render_mode="rgb_array")
env = PostureResidualWrapper(env, target)

model = load_latest_checkpoint(env)

if model is None:
    model = PPO.load("humanoid_posture_rl")

obs, _ = env.reset()

frames = []
action_history = []
q_error_history = []
qpos_history = []

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    frames.append(env.render())

    data = env.unwrapped.data
    qpos = data.qpos.copy()
    q = qpos[7:]
    q_target = target.qpos_target[7:]

    action_history.append(action)
    qpos_history.append(qpos)
    q_error_history.append(np.linalg.norm(q - q_target))

    if terminated or truncated:
        print(f"✅ 転倒 at step {step}")
        break

env.close()


# =====================
# GIF保存
# =====================
imageio.mimsave("humanoid_posture.gif", frames, fps=30)


# =====================
# データ整形
# =====================
qpos_history = np.array(qpos_history)
q_history = qpos_history[:, 7:]
q_target = target.qpos_target[7:]
action_history = np.array(action_history)


# =====================
# 誤差プロット
# =====================
plt.figure()
plt.plot(q_error_history)
plt.title("Posture Tracking Error")
plt.savefig("posture_error.png")
plt.close()


# =====================
# qpos
# =====================
plt.figure()
for i in range(q_history.shape[1]):
    plt.plot(q_history[:, i], label=f"qpos_{i}")
    plt.hlines(q_target[i], 0, len(q_history), linestyles='dashed')

plt.title("Tracking vs Target")
plt.legend()
plt.savefig("qpos_tracking.png")
plt.close()


# =====================
# action
# =====================
plt.figure()
for i in range(action_history.shape[1]):
    plt.plot(action_history[:, i], label=f"action_{i}")

plt.title("Action")
plt.legend()
plt.savefig("action.png")
plt.close()