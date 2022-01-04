from pathlib import Path
import sys
from scipy.stats import ttest_ind_from_stats

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import numpy as np

fontsize = 10
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

plt.subplots(1, 1, figsize=(2, 5))
lines = [x.split(", ") for x in open(r"figuresAndStats\segmentationFigure\data\ious.csv", "r").read().split("\n")][:-1]

names = [line[0] for line in lines]
ious = [[float(x) for x in line[1:]] for line in lines]

namesFormatted = [name + "\n(n=%d)" % len(iou) for name, iou in zip(names, ious)]

boxes = plt.boxplot(ious, patch_artist=True, zorder=0, widths=0.5)

for median in boxes['medians']:
    median.set_color('k')

colors = [(0, 205, 108)]
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
for patch in boxes['boxes']:
    patch.set_facecolor(colors[0])

for i, iou in enumerate(ious):
    plt.scatter([i + 1 for _ in iou], iou, marker='o', color='k', s=10, zorder=3)
    print()

plt.axhline(0.5, color=colors[0], ls='dashed')
plt.ylim([0.4, 1])
plt.xticks([1, 2, 3, 4], namesFormatted)
plt.ylabel("Intersection-over-union")

names.insert(1, "Non-PDAC")
ious.insert(1, sum(ious[1:], []))

print("Summary statistics:")
for name, iou in zip(names, ious):
    print("\t%s: %f (SD=%f)" % (name, np.mean(iou), np.std(iou)))

print("Statistical testing:")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = ious[i]
        b = ious[j]
        stat, p = ttest_ind_from_stats(np.mean(a), np.std(a), np.size(a),
                                       np.mean(b), np.std(b), np.size(b), equal_var=False)
        print("%s - %s: %s" % (names[i], names[j], str(p)))

plt.show()
