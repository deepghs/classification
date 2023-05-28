import numpy as np
from scipy import integrate
from sklearn.metrics import precision_recall_curve

from classification.plot.pr import nm_check


def cls_map_score(y_true, y_scores, labels):
    N, M = nm_check(y_true, y_scores, labels)
    values = []
    for mi in range(M):
        _y_true = y_true == mi
        _y_score = y_scores[:, mi]
        precision, recall, _ = precision_recall_curve(_y_true, _y_score)
        values.append(-integrate.simpson(precision, recall))

    return np.concatenate(values).mean()
