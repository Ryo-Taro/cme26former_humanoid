import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


# =====================
# 設定
# =====================
num_envs = 64
total_timesteps = 20_000_000_000
n_steps = 1024
n_save = 50_000_000


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
# Checkpoint + Eval
# =====================
class CheckpointEvalCallback(BaseCallback):
    def __init__(self, eval_freq, checkpoint_dir, fig_dir):
        super().__init__()
        self.eval_freq = eval_freq
        self.checkpoint_dir = checkpoint_dir
        self.fig_dir = fig_dir
        self.last_eval = 0

    def _on_step(self):
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            print(f"\n✅ Eval at step {self.num_timesteps}")
            self.last_eval = self.num_timesteps
            self.run_eval(tag=str(self.num_timesteps))
        return True

    def _on_training_end(self):
        print("\n✅ Final Evaluation")
        self.run_eval(tag="final")

    def run_eval(self, tag):

        # =====================
        # checkpoint保存
        # =====================
        model_path = os.path.join(self.checkpoint_dir, f"model_{tag}")
        self.model.save(model_path)

        # =====================
        # 評価環境
        # =====================
        env = gym.make("Humanoid-v5", render_mode="rgb_array")
        env = SimpleRewardWrapper(env)

        obs, _ = env.reset()

        frames = []
        qpos_history = []
        qvel_history = []
        action_history = []

        for step in range(1000):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            frames.append(env.render())

            data = env.unwrapped.data
            qpos_history.append(data.qpos.copy())
            qvel_history.append(data.qvel.copy())
            action_history.append(action)

            if terminated or truncated:
                print(f"✅ 転倒 at step {step}")
                break

        env.close()

        # =====================
        # GIF保存
        # =====================
        gif_path = os.path.join(self.fig_dir, f"eval_{tag}.gif")
        imageio.mimsave(gif_path, frames, fps=30)

        # =====================
        # データ整形
        # =====================
        qpos_history = np.array(qpos_history)
        qvel_history = np.array(qvel_history)
        action_history = np.array(action_history)

        q = qpos_history[:, 7:]
        qd = qvel_history[:, 6:]

        # =====================
        # qpos
        # =====================
        plt.figure()
        for i in range(q.shape[1]):
            plt.plot(q[:, i])
        plt.title(f"qpos_{tag}")
        plt.savefig(os.path.join(self.fig_dir, f"qpos_{tag}.png"))
        plt.close()

        # =====================
        # qvel
        # =====================
        plt.figure()
        for i in range(qd.shape[1]):
            plt.plot(qd[:, i])
        plt.title(f"qvel_{tag}")
        plt.savefig(os.path.join(self.fig_dir, f"qvel_{tag}.png"))
        plt.close()

        # =====================
        # action
        # =====================
        plt.figure()
        for i in range(action_history.shape[1]):
            plt.plot(action_history[:, i])
        plt.title(f"action_{tag}")
        plt.savefig(os.path.join(self.fig_dir, f"action_{tag}.png"))
        plt.close()


# =====================
# Reward Wrapper
# =====================
class SimpleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        q = qpos[7:]
        qd = qvel[6:]
        height = qpos[2]

        reward = 0.0
        reward -= 0.1 * np.sum(q**2)
        reward += 0.1 * height
        reward -= 0.000001 * np.sum(qd**2)
        reward += 0.5

        if height < 0.8:
            reward -= 20.0
            terminated = True

        return obs, reward, terminated, truncated, info


# =====================
# env生成
# =====================
def make_env():
    def _init():
        env = gym.make("Humanoid-v5")
        env = SimpleRewardWrapper(env)
        return env
    return _init


# =====================
# train
# =====================
def train():

    # =====================
    # Logsフォルダ作成
    # =====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("Logs", timestamp)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    fig_dir = os.path.join(log_dir, "figs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # =====================
    # 環境
    # =====================
    env_train = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # =====================
    # モデル
    # =====================
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=4096,
        gamma=0.995,
        clip_range=0.2,
        ent_coef=0.05,
    )

    # =====================
    # Callback
    # =====================
    callback1 = RewardLoggerCallback()

    callback2 = CheckpointEvalCallback(
        eval_freq=n_save,
        checkpoint_dir=checkpoint_dir,
        fig_dir=fig_dir,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback1, callback2],
    )

    env_train.close()

    # =====================
    # 学習曲線
    # =====================
    plt.figure()
    plt.plot(callback1.episode_rewards)
    plt.title("Training Reward")
    plt.savefig(os.path.join(fig_dir, "training_reward.png"))
    plt.close()


# =====================
# main
# =====================
if __name__ == "__main__":
    train()