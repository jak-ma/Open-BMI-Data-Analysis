# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter
# from scipy.signal import welch
# from sklearn.decomposition import PCA

# plt.rcParams["figure.dpi"] = 120

# def eda(X, y, fs=250, subject_name="subject", out_dir=None, show=True):
#     """
#     X: np.ndarray, (trials, channels, samples)
#     y: np.ndarray, (trials,)
#     fs: int, 采样率（默认 250）
#     subject_name: str, 用于文件名/标题
#     out_dir: None or path -> 若提供则保存图到该目录
#     show: whether to call plt.show()
#     """
#     if out_dir is not None:
#         os.makedirs(out_dir, exist_ok=True)

#     # ---------- 基本检查 ----------
#     n_trials, n_ch, n_samps = X.shape
#     print(f"[INFO] {subject_name}: trials={n_trials}, channels={n_ch}, samples={n_samps}, fs={fs}")

#     # ---------- 标签分布 ----------
#     cnt = Counter(y)
#     print(f"[INFO] Label counts: {cnt}")
#     labels, counts = zip(*sorted(cnt.items()))
#     total = sum(counts)
#     for lab, c in zip(labels, counts):
#         print(f"  label={lab}: count={c}, proportion={c/total:.3f}")

#     # ---------- 每 trial 总体幅值与方差分布 ----------
#     # per-trial mean abs amplitude and variance (across all channels & times)
#     trial_mean_abs = np.mean(np.abs(X), axis=(1,2))    # (n_trials,)
#     trial_var = np.var(X, axis=(1,2))

#     print(f"[INFO] Trial mean(|x|): mean={trial_mean_abs.mean():.4f}, std={trial_mean_abs.std():.4f}, min={trial_mean_abs.min():.4f}, max={trial_mean_abs.max():.4f}")
#     print(f"[INFO] Trial variance: mean={trial_var.mean():.6f}, std={trial_var.std():.6f}, min={trial_var.min():.6f}, max={trial_var.max():.6f}")

#     # plot trial-level histograms
#     plt.figure(figsize=(6,3))
#     plt.hist(trial_mean_abs, bins=40)
#     plt.title(f"{subject_name} - Trial mean(|x|) distribution")
#     plt.xlabel("mean(|x|)")
#     plt.ylabel("count")
#     if out_dir:
#         plt.savefig(os.path.join(out_dir, f"{subject_name}_trial_meanabs_hist.png"))
#     if show: plt.show()
#     plt.close()

#     plt.figure(figsize=(6,3))
#     plt.hist(trial_var, bins=40)
#     plt.title(f"{subject_name} - Trial variance distribution")
#     plt.xlabel("variance")
#     plt.ylabel("count")
#     if out_dir:
#         plt.savefig(os.path.join(out_dir, f"{subject_name}_trial_var_hist.png"))
#     if show: plt.show()
#     plt.close()

#     # ---------- 每通道统计（均值、标准差、方差） ----------
#     mean_per_ch = X.mean(axis=(0,2))   # (n_ch,)
#     std_per_ch  = X.std(axis=(0,2))
#     var_per_ch  = X.var(axis=(0,2))

#     print(f"[INFO] Per-channel mean: mean={mean_per_ch.mean():.6f}, std={mean_per_ch.std():.6f}")
#     print(f"[INFO] Per-channel std:  mean={std_per_ch.mean():.6f}, std={std_per_ch.std():.6f}")

#     # bar plot per-channel std (常用于发现坏通道)
#     plt.figure(figsize=(10,3))
#     plt.bar(np.arange(n_ch), std_per_ch)
#     plt.xlabel("channel")
#     plt.ylabel("std")
#     plt.title(f"{subject_name} - Per-channel std (possible bad channels have very small std or outliers)")
#     if out_dir:
#         plt.savefig(os.path.join(out_dir, f"{subject_name}_per_channel_std.png"))
#     if show: plt.show()
#     plt.close()

#     # ---------- 按标签分组比较（每通道均值 & std） ----------
#     uniq = np.unique(y)
#     plt.figure(figsize=(10,4))
#     for lab in uniq:
#         idx = np.where(y == lab)[0]
#         if len(idx) == 0:
#             continue
#         mean_ch_lab = X[idx].mean(axis=(0,2))
#         plt.plot(mean_ch_lab, label=f"label {lab}")
#     plt.xlabel("channel")
#     plt.ylabel("mean amplitude")
#     plt.legend()
#     plt.title(f"{subject_name} - Per-channel mean by label")
#     if out_dir:
#         plt.savefig(os.path.join(out_dir, f"{subject_name}_per_channel_mean_by_label.png"))
#     if show: plt.show()
#     plt.close()

#     # ---------- 简单的频域对比（每类平均 PSD） ----------
#     # 计算每 trial 的平均时序（所有通道平均），再计算 Welch PSD（也可以改为每通道）
#     # 这里做每类的 mean signal 的 PSD
#     plt.figure(figsize=(6,4))
#     for lab in uniq:
#         idx = np.where(y == lab)[0]
#         if len(idx) == 0:
#             continue
#         # 合并所有 trial 与通道，得到平均时间序列
#         mean_signal = X[idx].mean(axis=(0,1))   # shape=(n_samps,)
#         f, Pxx = welch(mean_signal, fs=fs, nperseg=min(1024, n_samps))
#         plt.semilogy(f, Pxx, label=f"label {lab}")
#     plt.xlim(0,60)
#     plt.xlabel("Hz")
#     plt.ylabel("PSD")
#     plt.title(f"{subject_name} - Mean-signal PSD by label")
#     plt.legend()
#     if out_dir:
#         plt.savefig(os.path.join(out_dir, f"{subject_name}_psd_by_label.png"))
#     if show: plt.show()
#     plt.close()

#     # ---------- PCA（用每 trial 通道均值作为特征），看类间分布 ----------
#     # feature: per-trial, per-channel mean -> shape (n_trials, n_ch)
#     feat = X.mean(axis=2)   # (n_trials, n_ch)
#     # 若通道数很大也可先降维或选通道
#     pca = PCA(n_components=2)
#     feat2 = pca.fit_transform(feat)
#     plt.figure(figsize=(5,4))
#     for lab in uniq:
#         idx = np.where(y==lab)[0]
#         plt.scatter(feat2[idx,0], feat2[idx,1], label=f"label {lab}", s=10, alpha=0.7)
#     plt.legend()
#     plt.title(f"{subject_name} - PCA on per-channel mean (2D)")
#     if out_dir:
#         plt.savefig(os.path.join(out_dir, f"{subject_name}_pca_per_channel_mean.png"))
#     if show: plt.show()
#     plt.close()

#     # ---------- 极端值 & 异常 trial 提醒 ----------
#     # 找出 trial variance 超出总体均值 +/- 3*std 的 trial
#     thr_hi = trial_var.mean() + 3 * trial_var.std()
#     thr_lo = trial_var.mean() - 3 * trial_var.std()
#     outlier_idx = np.where((trial_var > thr_hi) | (trial_var < thr_lo))[0]
#     print(f"[WARN] Outlier trials by variance (|> mean+3std|): {outlier_idx.tolist()} (count={len(outlier_idx)})")

#     # ---------- 返回 summary 字典 ----------
#     summary = {
#         "n_trials": n_trials,
#         "n_channels": n_ch,
#         "n_samples": n_samps,
#         "label_counts": dict(cnt),
#         "trial_mean_abs_stats": (float(trial_mean_abs.mean()), float(trial_mean_abs.std())),
#         "trial_var_stats": (float(trial_var.mean()), float(trial_var.std())),
#         "per_channel_mean": mean_per_ch,
#         "per_channel_std": std_per_ch,
#         "outlier_trial_idx": outlier_idx.tolist()
#     }

#     return summary

import numpy as np
import matplotlib.pyplot as plt

def plot_eeg_channel(X, trial_idx=0, ch_idx=0):
    """
    简单明了的 EDA：展示一个 trial 一个 channel 的
    1) 时间序列波形
    2) 直方图分布
    """

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

