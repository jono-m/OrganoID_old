import sys

from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from figuresAndStats.stats import pearsonr_ci, linr_ci
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

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
plt.subplot(1, 2, 1)
plt.plot(manual_counts, organoID_counts, 'o')
plt.title("Organoid counting comparison\n($CCC=%.2f$ [95%% CI %.2f-%.2f), $r=%.2f$ [95%% CI %.2f-%.2f]" % (
    ccc, loc, hic, r, lo, hi))
plt.ylabel("Number of organoids (Method: OrganoID)")
plt.xlabel("Number of organoids (Method: Manual)")
plt.plot([0, maxCount], [0, maxCount], "-")

plt.subplot(1, 2, 2)
means = [(x + y) / 2 for (x, y) in zip(organoID_counts, manual_counts)]
differences = [(x - y) for (x, y) in zip(organoID_counts, manual_counts)]
plt.plot(means, differences, 'o')

mean = np.mean(differences)
std = np.std(differences)
labels = [(mean, "Mean=%.2f" % mean, "b"),
          (mean - std * 1.96, "-1.96\u03C3=%.2f" % (mean - std * 1.96), "r"),
          (mean + std * 1.96, "+1.96\u03C3=%.2f" % (mean + std * 1.96), "r")]
plt.axhline(y=mean, color="b", linestyle="solid")
plt.axhline(y=mean + std * 1.96, color="r", linestyle="dashed")
plt.axhline(y=mean - std * 1.96, color="r", linestyle="dashed")
plt.xlabel("Average of OrganoID and manual count")
plt.ylabel("Difference between OrganoID and manual count")
plt.title("Bland-Altman plot of OrganoID and manual organoid count")
plt.ylim([min(differences) - 2, -min(differences) + 2])
for (y, text, color) in labels:
    plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right', color=color)

plt.show()
