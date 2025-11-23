import numpy as np
from mne.decoding import CSP

def apply_csp(X, y, n_components=4):
    """
    X: (n_trials, n_channels, n_times)
    y: (n_trials,)
    return:
        X_csp: (n_trials, n_components, n_times)
        csp: Model
    """
    csp = CSP(n_components=n_components, reg=None, log=False, norm_trace=False)
    X_csp = csp.fit_transform(X, y)  # (n_trials, n_components, n_times)

    return X_csp, csp
