import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import t


def MonteCarloIOU(n, p1, p2, size):
    aCalled = np.random.binomial(n, p1, size)
    bCalled = np.random.binomial(n, p1, size)
    intersect = np.random.binomial(np.max([aCalled, bCalled], axis=0), p2)
    union = aCalled + bCalled - intersect
    iou = intersect / union
    return iou


sizes = [100000]
imageSize = 512 * 512

ious = [MonteCarloIOU(imageSize, 0.2, 0.85, size) for size in sizes]
xs = [np.linspace(min(iou), max(iou), size) for size, iou in zip(sizes, ious)]
norms = [t.pdf(x, np.size(iou),
                  loc=np.mean(iou),
                  scale=np.std(iou)) for x, iou in zip(xs, ious)]

for iou, x, norm, size in zip(ious, xs, norms, sizes):
    seaborn.kdeplot(iou)
    plt.plot(x, norm)
    plt.legend(["IOU", "Normal (u=mean IOU, std=std IOU"])

plt.show()
