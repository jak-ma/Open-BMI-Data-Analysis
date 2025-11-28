import numpy as np
from mne.decoding import CSP
import mne 
### 禁用 INFO 打印
mne.set_log_level('WARNING')

def apply_csp(X, y, n_components=8):  
    csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y) 

    return X_csp, csp