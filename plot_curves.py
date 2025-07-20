import pandas as pd
import matplotlib.pyplot as plt
import os

# Map experiment paths to readable labels
experiments = {
    r"C:\Users\Ngum\Downloads\reworkk\DQN_learning\results\set1_cnnpolicy": "Set 1 (CNN)",
    r"C:\Users\Ngum\Downloads\reworkk\DQN_learning\results\set2_cnnpolicy": "Set 2 (CNN)",
    r"C:\Users\Ngum\Downloads\reworkk\DQN_learning\results\set3_cnnpolicy": "Set 3 (CNN)",
    r"C:\Users\Ngum\Downloads\reworkk\DQN_learning\results\set4_cnnpolicy": "Set 4 (CNN)",
    r"C:\Users\Ngum\Downloads\reworkk\DQN_learning\results\set1_mlppolicy": "Set 1 (MLP)"
}

plt.figure(figsize=(14, 8))

best_final_reward = float('-inf')
best_label = ""
best_timestep = 0

for exp, label in experiments.items():
    csv_path = os.path.join(exp, "training_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Optional: smooth the curve using rolling mean
        smoothed = df["mean_reward"].rolling(window=10, min_periods=1).mean()
        plt.plot(df["timestep"], smoothed, label=label, linewidth=2)
        # Track the best performing model
        if smoothed.iloc[-1] > best_final_reward:
            best_final_reward = smoothed.iloc[-1]
            best_label = label
            best_timestep = df["timestep"].iloc[-1]
    else:
        print(f"File not found: {csv_path}")

plt.xlabel("Timestep", fontsize=14)
plt.ylabel("Mean Reward (last 100 episodes)", fontsize=14)
plt.title("DQN Training Curves on Atari Assault", fontsize=16)
plt.legend(title="Experiments", fontsize=12, title_fontsize=13)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Annotate the best performing model
if best_label:
    plt.annotate(f"Best: {best_label}\nFinal Mean Reward: {best_final_reward:.1f}",
                 xy=(best_timestep, best_final_reward),
                 xytext=(best_timestep, best_final_reward + 10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, color='black', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.show()