import sys

from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from figuresAndStats.stats import pearsonr_ci, linr_ci
import numpy as np
import matplotlib.pyplot as plt

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
meanColor = [x / 255 for x in (0, 154, 222)]
lodColor = [x / 255 for x in (255, 31, 91)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

file = open(r"figuresAndStats\singleOrganoidFigure\data\counts.csv", "r")
lines = [line.split(", ") for line in file.read().split("\n")[1:-1]]
counts = [(float(line[1]), float(line[2])) for line in lines]

manual_counts = np.asarray([float(line[1]) for line in lines])
organoID_counts = np.asarray([float(line[2]) for line in lines])

correction = (len(organoID_counts) - 1) / len(organoID_counts)
covariance = np.cov(organoID_counts, manual_counts)[0, 1] * correction

ccc, loc, hic = linr_ci(organoID_counts, manual_counts)
r, p, lo, hi = pearsonr_ci(organoID_counts, manual_counts)

maxCount = np.max(np.concatenate([organoID_counts, manual_counts]))
plt.subplots(1, 2, figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.scatter(manual_counts, organoID_counts, marker='o', s=5, color='k')
plt.ylabel("Manual count")
plt.xlabel("OrganoID count")
plt.text(0, maxCount - 20, "$CCC=%.2f$\n[%.2f-%.2f]" % (ccc, loc, hic), verticalalignment='bottom', size=fontsize,
         color=corrColor)
plt.plot([0, maxCount], [0, maxCount], "-", color=corrColor)

plt.subplot(1, 2, 2)
means = [(x + y) / 2 for (x, y) in zip(organoID_counts, manual_counts)]
differences = [(x - y) for (x, y) in zip(organoID_counts, manual_counts)]
plt.scatter(means, differences, marker='o', s=5, color='k')
plt.legend(["Organoid image"])

mean = np.mean(differences)
std = np.std(differences)
labels = [(mean, "Mean=%.2f" % mean, meanColor),
          (mean - std * 1.96, "-1.96\u03C3=%.2f" % (mean - std * 1.96), lodColor),
          (mean + std * 1.96, "+1.96\u03C3=%.2f" % (mean + std * 1.96), lodColor)]
plt.axhline(y=mean, color=meanColor, linestyle="solid")
plt.axhline(y=mean + std * 1.96, color=lodColor, linestyle="dashed")
plt.axhline(y=mean - std * 1.96, color=lodColor, linestyle="dashed")
plt.xlabel("Count average")
plt.ylabel("Count difference")
plt.ylim([min(differences) - 2, -min(differences) + 2])
for (y, text, color) in labels:
    plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right', color=color,
             size=fontsize)

plt.show()
