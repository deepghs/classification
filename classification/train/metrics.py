import numpy as np
from scipy import integrate
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from classification.plot.pr import nm_check


def cls_map_score(y_true, y_scores, labels):
    N, M = nm_check(y_true, y_scores, labels)
    values = []
    for mi in range(M):
        _y_true = y_true == mi
        _y_score = y_scores[:, mi]
        precision, recall, _ = precision_recall_curve(_y_true, _y_score)
        values.append(-integrate.simpson(precision, recall))

    return np.array(values).mean()


def cls_auc_score(y_true, y_scores, labels):
    N, M = nm_check(y_true, y_scores, labels)
    values = []
    for mi in range(M):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, mi], pos_label=mi)
        auc_value = auc(fpr, tpr)
        values.append(auc_value)

    return np.array(values).mean()
