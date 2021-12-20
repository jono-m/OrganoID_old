from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt

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

names = [name + "\n(n=%d)" % len(iou) for name, iou in zip(names, ious)]

boxes = plt.boxplot(ious, patch_artist=True, zorder=0, widths=0.5)

for median in boxes['medians']:
    median.set_color('k')

colors = [(175, 88, 186)]
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
for patch in boxes['boxes']:
    patch.set_facecolor(colors[0])

for i, iou in enumerate(ious):
    plt.scatter([i + 1 for _ in iou], iou, marker='o', color='k', s=10, zorder=3)
plt.xticks([1, 2, 3, 4], names)
plt.ylabel("Intersection-over-union")
plt.show()
