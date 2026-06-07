import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

# =====================
# 改良版制御
# =====================
def get_action(env):
    data = env.unwrapped.data

    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    # ---- 関節部分 ----
    qpos_act = qpos[7:]
    qvel_act = qvel[6:]

    # ---- 対称な目標姿勢 ----
    target = np.zeros_like(qpos_act)

    # ---- 右脚 ----
    target[0] = -0.25   # hip (前後)
    target[1] = 0.05    # hip (左右)
    target[3] = 0.5     # knee
    target[6] = -0.15   # ankle

    # ---- 左脚 ----
    target[7] = -0.25
    target[8] = -0.05
    target[10] = 0.5
    target[13] = -0.15


    # ---- 基本PD ----
    kp = 100.0
    kd = 8.0
    action = kp * (target - qpos_act) - kd * qvel_act

    # ---- COM制御 ----
    com = data.subtree_com[0]
    com_vel = data.cvel[0][:3]

    com_x = com[0]
    com_y = com[1]

    # ---- 前後補正 ----
    action[0] += -80.0 * com_x - 8.0 * com_vel[0]   # 右hip
    action[6] += -60.0 * com_x                      # 右ankle

    action[7] += -80.0 * com_x - 8.0 * com_vel[0]   # 左hip
    action[13] += -60.0 * com_x                     # 左ankle

    # ---- 左右補正（ここが追加ポイント）----
    action[1] += -60.0 * com_y   # 右hip lateral
    action[8] += -60.0 * com_y   # 左hip lateral

    # ---- トルク制限 ----
    return np.clip(action, -1.0, 1.0)


# =====================
# 実行
# =====================
env = gym.make("Humanoid-v5", render_mode="rgb_array")

obs, _ = env.reset()

frames = []

steps = 1000
survival_steps = 0

for step in range(steps):
    action = get_action(env)

    obs, reward, terminated, truncated, _ = env.step(action)

    frame = env.render()
    frames.append(frame)

    survival_steps = step

    if terminated or truncated:
        print(f"✅ 転倒 at step {step}")
        break

env.close()

# =====================
# GIF保存
# =====================
imageio.mimsave("humanoid.gif", frames, fps=30)

print(f"✅ 生存ステップ数: {survival_steps}")
print("✅ humanoid.gif 保存完了")

# =====================
# 高さプロット
# =====================
heights = [f[0][0] for f in frames]  # 簡易化

plt.plot(heights)
plt.title("Approx Height Trend")
plt.xlabel("Step")
plt.ylabel("Height-like")
plt.show()