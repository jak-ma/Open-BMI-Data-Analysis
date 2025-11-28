import numpy as np
import matplotlib.pyplot as plt

def plot_eeg_channel(X, trial_idx=0, ch_idx=0):

    signal = X[trial_idx, ch_idx]     # (samples,)
    samples = signal.shape[0]

    plt.figure(figsize=(14, 5))

    # 1. 时间波形
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title(f"Trial {trial_idx}, Channel {ch_idx} - Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    # 2. 直方图
    plt.subplot(1, 2, 2)
    plt.hist(signal, bins=50)
    plt.title(f"Trial {trial_idx}, Channel {ch_idx} - Histogram")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

def plot_csp_feature_hist(X_csp, feature_idx):
    plt.hist(X_csp[:, feature_idx], bins=30)
    plt.title(f"CSP feature {feature_idx}")
    plt.show()

