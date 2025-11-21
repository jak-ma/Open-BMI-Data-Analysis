import os
from pre_process import preprocess_raw_data

### Configs
# path
sess01_dir = 'input/sess01'
subj_paths = [f'sess01_subj{i:02d}_EEG_MI.mat' for i in range(1, 55)]



if __name__ == '__main__':
    X, y, _ = preprocess_raw_data(os.path.join(sess01_dir, subj_paths[0]), 'train')
    print(X.shape, y.shape)
    
    X, y, _ = preprocess_raw_data(os.path.join(sess01_dir, subj_paths[0]), 'test')
    print(X.shape, y.shape)


