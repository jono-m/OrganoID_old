import sys

from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from figuresAndStats.stats import pearsonr_ci, linr_ci
import numpy as np
import matplotlib.pyplot as plt
from backend.Detector import Detector
from backend.ImageManager import LoadImages
from backend.Label import Label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndimage

plt.rcParams['svg.fonttype'] = 'none'

modelPath = Path(r"model\model.tflite")
imagesPath = Path(r"dataset\testing\images")
segmentationsPath = Path(r"dataset\testing\segmentations")

detector = Detector(modelPath)

images = LoadImages(imagesPath, [512, 512], mode="L")
segmentations = LoadImages(segmentationsPath, [512, 512], mode="1")

organoID_counts = []
manual_counts = []

for (image, segmentation) in zip(images, segmentations):
    print("Analyzing image %s" % image.path.name)
    detected = detector.Detect(image.frames[0])
    organoID_labeled = Label(detected, 200, False)

    manual_labeled, _ = ndimage.label(segmentation.frames[0])
    manual_labeled = remove_small_objects(manual_labeled, 200)

    num_organoID = len(regionprops(organoID_labeled))
    num_manual = len(regionprops(manual_labeled))

    organoID_counts.append(num_organoID)
    manual_counts.append(num_manual)

organoID_counts = np.asarray(organoID_counts)
manual_counts = np.asarray(manual_counts)

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
