import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
from figuresAndStats.stats import pearsonr_ci, linr_ci
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'

file = open(r"figuresAndStats\singleOrganoidFigure\data\matched_areas.csv", "r")
lines = [line.split(", ") for line in file.read().split("\n")[1:-1]]
areas = [(float(line[2]), float(line[4])) for line in lines]

areas = np.asarray(areas) / 1000
organoID_areas = areas[:, 0]
manual_areas = areas[:, 1]

ccc, loc, hic = linr_ci(organoID_areas, manual_areas)

r, p, lo, hi = pearsonr_ci(organoID_areas, manual_areas)

maxArea = np.max(np.concatenate([organoID_areas, manual_areas]))
plt.subplot(1, 2, 1)
plt.plot(manual_areas, organoID_areas, 'o')
plt.title("Organoid area comparison\n($CCC=%.2f$ [95%% CI %.2f-%.2f], $r=%.2f$ [95%% CI %.2f-%.2f]" % (
    ccc, loc, hic, r, lo, hi))
plt.ylabel(r"Organoid area (method: OrganoID, x $10^3 \mu m^2$))")
plt.xlabel(r"Organoid area (method: manual, x $10^3 \mu m^2$))")
plt.plot([0, maxArea], [0, maxArea], "-")

plt.subplot(1, 2, 2)
means = [(x + y) / 2 for (x, y) in zip(organoID_areas, manual_areas)]
differences = [(x - y) for (x, y) in zip(organoID_areas, manual_areas)]
plt.plot(means, differences, 'o')

mean = np.mean(differences)
std = np.std(differences)
labels = [(mean, r"Mean=%.2f x $10^3 \mu m^2$" % mean, "b"),
          (mean - std * 1.96, "-1.96\u03C3=%.2f x $10^3 \mu m^2$" % (mean - std * 1.96), "r"),
          (mean + std * 1.96, "+1.96\u03C3=%.2f x $10^3 \mu m^2$" % (mean + std * 1.96), "r")]
plt.axhline(y=mean, color="b", linestyle="solid")
plt.axhline(y=mean + std * 1.96, color="r", linestyle="dashed")
plt.axhline(y=mean - std * 1.96, color="r", linestyle="dashed")
plt.xlabel(r"Average of OrganoID and manual area (x $10^3 \mu m^2$)")
plt.ylabel(r"Difference between OrganoID and manual area (x $10^3 \mu m^2$)")
plt.title("Bland-Altman plot of OrganoID and manual organoid area")
plt.ylim([min(differences) - 2, -min(differences) + 2])
for (y, text, color) in labels:
    plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right', color=color)
plt.show()
