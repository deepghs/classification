from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal


def plt_confusion_matrix(ax, y_true, y_pred, labels, title: str = 'Confusion Matrix',
                         normalize: Literal['true', 'pred', None] = None, cmap=None) -> Image.Image:
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    disp.plot(ax=ax, cmap=cmap or plt.cm.Blues)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.set_title(title)
