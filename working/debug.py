import os
from utils.pre_process import preprocess_raw_data
from utils.csp import apply_csp
from utils.eda import plot_eeg_eda, plot_csp_eda
import numpy as np


sess01_dir = 'input/sess01'
subj_paths = [f'sess01_subj{i:02d}_EEG_MI.mat' for i in range(1, 55)]

if __name__ == '__main__':
    X_train, y_train = preprocess_raw_data(os.path.join(sess01_dir, subj_paths[1]), 'train')
    
    print("Preprocessed EEG shape:", X_train.shape)  # (100, 20, 626)
    print("Labels:", np.unique(y_train, return_counts=True))

    # 1. 原始 EEG EDA
    # plot_eeg_eda(X_train, y_train, trial_idx=0)

    # 2. CSP 特征提取
    X_csp, _ = apply_csp(X_train, y_train)  # 应返回 (100, n_components)
    print("CSP features shape:", X_csp.shape)

    # 3. CSP EDA（以第0个特征为例）
    plot_csp_eda(X_csp, y_train, feature_idx=2)

