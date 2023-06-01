import numpy as np
from hbutils.random import keep_global_state
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay


def nm_check(y_true, y_scores, labels):
    assert len(y_true.shape) == 1, \
        f'y_true should be 1-dim, but {y_true.shape!r} found.'
    N = y_true.shape[0]
    M = len(labels)
    assert y_scores.shape == (N, M), \
        f'y_scores\' shape should be {(N, M)!r}, but {y_scores.shape!r} found.'

    return N, M


@keep_global_state()
def _create_score_curve(ax, name, func, y_true, y_scores, labels, title=None, units: int = 500):
    N, M = nm_check(y_true, y_scores, labels)
    for mi in range(M):
        xs, ys = [], []
        scores = np.sort(y_scores[:, mi], kind='heapsort')
        if len(scores) > units:
            scores = np.random.choice(scores, units)
        for score in np.sort(scores, kind='heapsort'):
            _y_true = y_true == mi
            _y_pred = y_scores[:, mi] >= score
            # f1_score()
            precision = func(_y_true, _y_pred, zero_division=1)
            xs.append(score)
            ys.append(precision)

        xs = np.array(xs)
        ys = np.array(ys)
        maxj = np.argmax(ys)
        ax.plot(xs, ys, label=f'{labels[mi]} ({ys[maxj]:.2f} at {xs[maxj]:.3f})')

    ax.set_xlabel(f'score')
    ax.set_ylabel(f'{name}')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title(title or f'{name} curve'.capitalize())
    ax.grid()
    ax.legend()


def plt_f1_curve(ax, y_true, y_scores, labels, title='F1 Curve', units: int = 500):
    _create_score_curve(ax, 'F1', f1_score, y_true, y_scores, labels, title, units)


def plt_p_curve(ax, y_true, y_scores, labels, title='Precision Curve', units: int = 500):
    _create_score_curve(ax, 'precision', precision_score, y_true, y_scores, labels, title, units)


def plt_r_curve(ax, y_true, y_scores, labels, title='Recall Curve', units: int = 500):
    _create_score_curve(ax, 'recall', recall_score, y_true, y_scores, labels, title, units)


def plt_pr_curve(ax, y_true, y_scores, labels, title='PR Curve'):
    N, M = nm_check(y_true, y_scores, labels)
    for mi in range(M):
        _y_true = y_true == mi
        _y_score = y_scores[:, mi]
        precision, recall, _ = precision_recall_curve(_y_true, _y_score)
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        _map = -np.trapz(precision, recall)
        disp.plot(ax=ax, name=f'{labels[mi]} (mAP {_map:.3f})')

    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid()
    ax.legend()
