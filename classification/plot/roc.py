from sklearn.metrics import roc_curve, auc, RocCurveDisplay


def plt_roc_curve(ax, y_true, y_score, labels, title: str = 'ROC Curve'):
    for i, label in enumerate(labels):
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:, i], pos_label=i)
        auc_value = auc(fpr, tpr)

        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_value, estimator_name=label)
        display.plot(ax=ax)

    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid()
    ax.legend()
