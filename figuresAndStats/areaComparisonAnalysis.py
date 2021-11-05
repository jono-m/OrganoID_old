import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from util.stats import pearsonr_ci, linr_ci

csvFile = open(r"C:\Users\jonoj\Google Drive\Research\OrganoID\Data\TestingData\SingleComparison\areas.csv", "r")
data = csvFile.read()
lines = data.split("\n")
lines = [line.split(", ") for line in lines]
lines = [line[1:] for line in lines]
lines = [[int(x) for x in y] for y in lines]


predictedAreas = [line for (lineNumber, line) in enumerate(lines) if lineNumber % 2 == 0][:-1]
manualAreas = [line for (lineNumber, line) in enumerate(lines) if lineNumber % 2 == 1]

predictedAreas = [area for frame in predictedAreas for area in frame]
manualAreas = [area for frame in manualAreas for area in frame]
areas = np.stack([predictedAreas, manualAreas]).transpose()
areas = areas[np.all(areas > 200, axis=1), :]

predictedAreas = areas[:, 0]
manualAreas = areas[:, 1]


ccc, loc, hic = linr_ci(predictedAreas, manualAreas)

r, p, lo, hi = pearsonr_ci(predictedAreas, manualAreas)

maxArea = np.max(np.concatenate([predictedAreas, manualAreas]))
plt.subplot(1, 2, 1)
plt.plot(manualAreas, predictedAreas, 'o')
plt.title("Organoid area comparison\n($CCC=%.2f$ [95%% CI %.2f-%.2f], $r^2=%.2f$ [95%% CI %.2f-%.2f]" % (ccc, loc, hic, r**2, lo**2, hi**2))
plt.ylabel("Organoid area (Method: OrganoID)")
plt.xlabel("Organoid area (Method: Manual)")
plt.plot([0, maxArea], [0, maxArea], "-")

plt.subplot(1, 2, 2)
means = [(x + y) / 2 for (x, y) in zip(predictedAreas, manualAreas)]
differences = [(x - y) for (x, y) in zip(predictedAreas, manualAreas)]
plt.plot(means, differences, 'o')

mean = np.mean(differences)
std = np.std(differences)
labels = [(mean, "Mean=%.2f" % mean, "b"),
          (mean - std * 1.96, "-1.96\u03C3=%.2f" % (mean - std * 1.96), "r"),
          (mean + std * 1.96, "+1.96\u03C3=%.2f" % (mean + std * 1.96), "r")]
plt.axhline(y=mean, color="b", linestyle="solid")
plt.axhline(y=mean + std * 1.96, color="r", linestyle="dashed")
plt.axhline(y=mean - std * 1.96, color="r", linestyle="dashed")
plt.xlabel("Average of OrganoID and manual area ($\mu m^2)")
plt.ylabel("Difference between OrganoID and manual area")
plt.title("Bland-Altman plot of OrganoID and manual organoid area")
plt.ylim([min(differences) - 2, -min(differences) + 2])
for (y, text, color) in labels:
    plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right', color=color)











plt.show()
