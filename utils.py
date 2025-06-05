import numpy as np
from sklearn.metrics import matthews_corrcoef

def encontrar_umbral_optimo_mcc(y_true, y_prob, n_steps=100):
    best_mcc = -1
    best_thresh = 0.5
    thresholds = np.linspace(0, 1, n_steps)
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_t)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
    return best_thresh, best_mcc