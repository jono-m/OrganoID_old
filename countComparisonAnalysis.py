import numpy as np
import matplotlib.pyplot as plt
from util.stats import pearsonr_ci, linr_ci

csvFile = open(r"C:\Users\jonoj\Documents\ML\SingleComparison\counts.csv", "r")
data = csvFile.read()
lines = data.split("\n")
lines = [line.split(", ") for line in lines]
lines = [line[1:] for line in lines]
lines = [[int(x) for x in y] for y in lines]

predictedAreas = [line for (lineNumber, line) in enumerate(lines) if lineNumber % 2 == 0][:-1]
manualAreas = [line for (lineNumber, line) in enumerate(lines) if lineNumber % 2 == 1]

cutoff = 200
predictedAreas = [[x for x in y if x > cutoff] for y in predictedAreas]
manualAreas = [[x for x in y if x > cutoff] for y in manualAreas]

predictedCounts = np.asarray([len(line) for line in predictedAreas])
manualCounts = np.asarray([len(line) for line in manualAreas])

correction = (len(predictedCounts)-1)/len(predictedCounts)
covariance = np.cov(predictedCounts, manualCounts)[0, 1] * correction

ccc, loc, hic = linr_ci(predictedCounts, manualCounts)
r, p, lo, hi = pearsonr_ci(predictedCounts, manualCounts)

maxCount = np.max(np.concatenate([predictedCounts, manualCounts]))
plt.subplot(1, 2, 1)
plt.plot(manualCounts, predictedCounts, 'o')
plt.title("Organoid counting comparison\n($CCC=%.2f$ [95%% CI %.2f-%.2f), $r^2=%.2f$ [95%% CI %.2f-%.2f]" % (ccc, loc, hic, r**2, lo**2, hi**2))
plt.ylabel("Number of organoids (Method: OrganoID)")
plt.xlabel("Number of organoids (Method: Manual)")
plt.plot([0, maxCount], [0, maxCount], "-")

plt.subplot(1, 2, 2)
means = [(x + y) / 2 for (x, y) in zip(predictedCounts, manualCounts)]
differences = [(x - y) for (x, y) in zip(predictedCounts, manualCounts)]
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
