import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
from figuresAndStats.stats import pearsonr_ci, linr_ci
import matplotlib.pyplot as plt
import numpy as np

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
meanColor = [x / 255 for x in (0, 154, 222)]
lodColor = [x / 255 for x in (0, 205, 108)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

file = open(r"figuresAndStats\singleOrganoidFigure\data\matched_areas.csv", "r")
lines = [line.split(", ") for line in file.read().split("\n")[1:-1]]
areas = [(float(line[2]), float(line[4])) for line in lines]

areas = np.asarray(areas) / 1000
organoID_areas = areas[:, 0]
manual_areas = areas[:, 1]

ccc, loc, hic = linr_ci(organoID_areas, manual_areas)

r, p, lo, hi = pearsonr_ci(organoID_areas, manual_areas)

maxArea = np.max(np.concatenate([organoID_areas, manual_areas]))
plt.subplots(1, 2, figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.scatter(manual_areas, organoID_areas, marker='o', color='k', s=5)
plt.ylabel(r"OrganoID-computed area (x $10^3 \mu m^2$)")
plt.xlabel(r"Manual-computed area (x $10^3 \mu m^2$))")
plt.text(0, maxArea - 20, "$CCC=%.2f$\n[%.2f-%.2f]" % (ccc, loc, hic), verticalalignment='bottom', size=fontsize,
         color=corrColor)
plt.plot([0, maxArea], [0, maxArea], "-", color=corrColor)

plt.subplot(1, 2, 2)
means = [(x + y) / 2 for (x, y) in zip(organoID_areas, manual_areas)]
differences = [(x - y) for (x, y) in zip(organoID_areas, manual_areas)]
plt.scatter(means, differences, marker='o', color='k', s=5)

mean = np.mean(differences)
std = np.std(differences)
labels = [(mean, r"Mean=%.2f x $10^3 \mu m^2$" % mean, meanColor),
          (mean - std * 1.96, "-1.96\u03C3=%.2f x $10^3 \mu m^2$" % (mean - std * 1.96), lodColor),
          (mean + std * 1.96, "+1.96\u03C3=%.2f x $10^3 \mu m^2$" % (mean + std * 1.96), lodColor)]
plt.axhline(y=mean, color=meanColor, linestyle="solid")
plt.axhline(y=mean + std * 1.96, color=lodColor, linestyle="dashed")
plt.axhline(y=mean - std * 1.96, color=lodColor, linestyle="dashed")
plt.xlabel(r"Area average (x $10^3 \mu m^2$)")
plt.ylabel(r"Area difference (x $10^3 \mu m^2$)")
plt.ylim([min(differences) - 2, -min(differences) + 2])
for (y, text, color) in labels:
    plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right', color=color,
             size=fontsize)
plt.show()
