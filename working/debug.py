import os
from utils.pre_process import preprocess_raw_data
from utils.csp import apply_csp
import numpy as np
from sklearn.preprocessing import PowerTransformer

sess01_dir = 'input/sess01'
subj_paths = [f'sess01_subj{i:02d}_EEG_MI.mat' for i in range(1, 55)]

if __name__ == '__main__':
    X, y = preprocess_raw_data(os.path.join(sess01_dir, subj_paths[0]), 'train')
    print(X.shape)
    # print(X.shape, y.shape)
    # eda(X, y)
    # plot_eeg_channel(X, 1, 1)
    X_csp, _ = apply_csp(X, y)
    
    # plot_csp_feature_hist(X_csp, 1)

    # X_csp = np.log1p(X_csp)
    print(X_csp.shape)
    # plot_csp_feature_hist(X_csp, 1)

    # plot_csp_feature_hist(X_csp, 1)
    # print(X_csp.shape, y.shape)
    # eda(X_csp, y)


