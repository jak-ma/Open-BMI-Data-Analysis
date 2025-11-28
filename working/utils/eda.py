import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def plot_eeg_eda(X, y=None, trial_idx=0):
    """
    对原始 EEG 数据做 EDA
    X: (n_trials, n_channels, n_times)
    y: (n_trials,) optional, for coloring
    """
    n_trials, n_chans, n_times = X.shape
    
    # 1. 随机选一个 trial，画所有通道的时间序列
    plt.figure(figsize=(15, 8))
    for ch in range(n_chans):
        plt.plot(X[trial_idx, ch] + ch * 10, label=f'Ch {ch}')  # 垂直偏移便于观察
    plt.title(f"EEG Time Series - Trial {trial_idx} (each channel offset by 10μV)")
    plt.xlabel("Time Sample (0–625 @ 250 Hz → 0–2.5s)")
    plt.ylabel("Amplitude (μV) + Channel Offset")
    plt.tight_layout()
    plt.show()

    # 2. 全局 amplitude 分布（所有 trials & channels）
    all_vals = X.flatten()
    plt.figure(figsize=(10, 4))
    plt.hist(all_vals, bins=100, alpha=0.7, color='skyblue')
    plt.title("Distribution of All EEG Amplitudes (across trials & channels)")
    plt.xlabel("Amplitude (μV)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    print(f"EEG stats: mean={all_vals.mean():.3f}, std={all_vals.std():.3f}")


def plot_csp_eda(X_csp, y, feature_idx=0):
    """
    对 CSP 特征做 EDA
    X_csp: (n_trials, n_features)
    y: (n_trials,) with labels 0/1
    """
    # 1. 原始 CSP 特征分布（按类别着色）
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 1, 1)
    for cls in np.unique(y):
        mask = (y == cls)
        plt.hist(X_csp[mask, feature_idx], bins=30, alpha=0.6, 
                 label=f'Class {cls}', density=True)
    plt.title(f"CSP Feature {feature_idx} Distribution (Raw)")
    plt.xlabel("Feature Value (variance after spatial filtering)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    