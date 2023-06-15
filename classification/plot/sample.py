import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from hbutils.system import TemporaryDirectory
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_samples(y_true, y_pred, visuals, concen_cls: int, labels: List[str],
                 samples_per_case: int = 10, figsize=None):
    figsize = figsize or (samples_per_case * 0.85, (len(labels) + 1) * 0.9)
    fig = plt.figure(figsize=figsize)
    try:
        fig.tight_layout()
        grid = ImageGrid(fig, 111, nrows_ncols=(len(labels), samples_per_case), axes_pad=0, share_all=True)

        total = (y_true == concen_cls).sum()
        for xi, row in enumerate(grid.axes_row):
            ids, = np.where((y_true == concen_cls) & (y_pred == xi))
            cnt = ids.shape[0]
            if cnt == 0:
                cases = []
            else:
                ids = ids[:samples_per_case]
                cases = (visuals[ids, ...].transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
                cases = list(cases)
            if len(cases) < samples_per_case:
                cases += [None] * (samples_per_case - len(cases))

            cur = ((y_true == concen_cls) & (y_pred == xi)).sum()
            for yi, a in enumerate(row):
                if cases[yi] is not None:
                    a.imshow(cases[yi])
                    a.set_xticklabels([])
                    a.set_yticklabels([])

                if yi == 0:
                    a.set_ylabel(
                        f'{labels[xi]}\n({cur}, {cur * 100.0 / total:.1f}%)',
                        fontweight='bold' if concen_cls == xi else None,
                        fontsize=7,
                    )

        correct = ((y_true == concen_cls) & (y_pred == concen_cls)).sum()
        acc = correct * 1.0 / total
        fig.suptitle(f'Predictions of {labels[concen_cls]}\n'
                     f'total: {total}, correct: {correct}, acc: {acc:.4f}')

        with TemporaryDirectory() as td:
            imgfile = os.path.join(td, 'image.png')
            fig.savefig(imgfile)

            image = Image.open(imgfile)
            image.load()
            image = image.convert('RGB')
            return image

    finally:
        plt.close(fig)