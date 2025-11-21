import numpy as np
import scipy.io as sio
import mne

def preprocess_raw_data(mat_path, type_str):
    # global value
    chs = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
    
    # load raw data
    data0 = sio.loadmat(mat_path)
    
    use_type_str = f'EEG_MI_{type_str}'
    data = data0[use_type_str][0, 0]
    print("Raw x shape in .mat:", data['x'].shape)

    # extract EEG -> (ch, time)
    EEG = data['x'].T.astype(np.float64)
    fs = float(data['fs'])
    ch_names = [c[0] for c in data['chan'][0]]

    # create MNE Info And Raw 
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(EEG, info, verbose=False)

    # get events: cue time + label
    t_cues = data['t'].flatten() - 1
    y = data['y_dec'].flatten()
    # MNE 要求event_id 不能为0/1 -> 10/20
    event_id = {'left':10, 'right':20}
    events = np.column_stack([
        t_cues.astype(int),
        np.zeros_like(t_cues, dtype=int),
        np.where(y == 1, 10, 20)
    ])

    # split 
    epochs = mne.Epochs(raw, events, event_id,
                    tmin=1.0, tmax=3.5,
                    baseline=None, preload=True, verbose=False)
    
    # choose 20 chs
    epochs.pick_channels(chs)
    # filter (8-30 Hz)
    epochs.filter(l_freq=8, h_freq=30, verbose=False)

    X = epochs.get_data() # (n_trials, n_ch, n_time)
    y = np.where(epochs.events[:, 2] == 10, 1, 2)
    print(fs)
    return X, y, {'fs':fs, 'ch_names':chs, 'tmin':1.0, 'tmax':3.5, 'n_trials':len(X)}