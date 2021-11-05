import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
from figuresAndStats.stats import pearsonr_ci, linr_ci
from backend.Detector import Detector
from backend.ImageManager import LoadImages, LabelToRGB, ShowImage
from backend.Label import Label
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndimage

plt.rcParams['svg.fonttype'] = 'none'

# Set this to TRUE to display the matched organoids between OrganoID- and manual-labeled
# images.
displayComparison = False


def Relabel(a: np.ndarray, b: np.ndarray, labelPairs):
    newA = np.zeros_like(a)
    newB = np.zeros_like(b)

    for (i, (aLabel, bLabel)) in enumerate(labelPairs):
        newA[a == aLabel] = i
        newB[b == bLabel] = i
    return newA, newB


modelPath = Path(r"model\model.tflite")
imagesPath = Path(r"dataset\testing\images")
segmentationsPath = Path(r"dataset\testing\segmentations")

detector = Detector(modelPath)

images = LoadImages(imagesPath, [512, 512], mode="L")
segmentations = LoadImages(segmentationsPath, [512, 512], mode="1")

organoidAreas = []

for (image, segmentation) in zip(images, segmentations):
    print("Analyzing image %s " % image.path.name)
    detected = detector.Detect(image.frames[0])
    organoID_labeled = Label(detected, 200, False)

    manual_labeled, _ = ndimage.label(segmentation.frames[0])
    manual_labeled = remove_small_objects(manual_labeled, 200)

    organoIDOrganoids = regionprops(organoID_labeled)
    manualOrganoids = regionprops(manual_labeled)

    # Now we match organoids in the manual and OrganoID labeled images by POSITION only
    # (to compare area). Use Hungarian assignment algorithm with cost matrix of centroid
    # distance.
    numPredicted = len(organoIDOrganoids)
    numActual = len(manualOrganoids)
    fullSize = max(numPredicted, numActual)
    costMatrix = np.zeros([fullSize, fullSize])
    for predictedIndex, predictedOrganoid in enumerate(organoIDOrganoids):
        for manualIndex, manualOrganoid in enumerate(manualOrganoids):
            distance = np.sqrt((predictedOrganoid.centroid[0] - manualOrganoid.centroid[0]) ** 2 +
                               (predictedOrganoid.centroid[1] - manualOrganoid.centroid[1]) ** 2)
            costMatrix[predictedIndex, manualIndex] = distance
    predictedIndices, manualIndices = linear_sum_assignment(costMatrix)

    # Gather areas for the position-matched organoids
    pairCount = min(numPredicted, numActual)
    pairs = [pair for (i, pair) in enumerate(zip(predictedIndices, manualIndices)) if
             pair[0] < pairCount and pair[1] < pairCount]
    labelPairs = [(organoIDOrganoids[predictedIndex].label, manualOrganoids[manualIndex].label)
                  for (predictedIndex, manualIndex) in pairs]
    areaPairs = [(organoIDOrganoids[predictedIndex].area, manualOrganoids[manualIndex].area)
                 for (predictedIndex, manualIndex) in pairs]

    organoidAreas += areaPairs

    if displayComparison:
        newA, newB = Relabel(organoID_labeled, manual_labeled, labelPairs)
        aImage = LabelToRGB(newA, True)
        bImage = LabelToRGB(newB, True)

        mergedA = np.concatenate([LabelToRGB(organoID_labeled, True), LabelToRGB(manual_labeled, True)], axis=1)
        mergedB = np.concatenate([aImage, bImage], axis=1)
        merged = np.concatenate([mergedA, mergedB])

        ShowImage(merged)
        input()

areas = np.asarray(organoidAreas)
organoID_areas = areas[:, 0]
manual_areas = areas[:, 1]

ccc, loc, hic = linr_ci(organoID_areas, manual_areas)

r, p, lo, hi = pearsonr_ci(organoID_areas, manual_areas)

maxArea = np.max(np.concatenate([organoID_areas, manual_areas]))
plt.subplot(1, 2, 1)
plt.plot(manual_areas, organoID_areas, 'o')
plt.title("Organoid area comparison\n($CCC=%.2f$ [95%% CI %.2f-%.2f], $r=%.2f$ [95%% CI %.2f-%.2f]" % (
    ccc, loc, hic, r, lo, hi))
plt.ylabel("Organoid area (Method: OrganoID)")
plt.xlabel("Organoid area (Method: Manual)")
plt.plot([0, maxArea], [0, maxArea], "-")

plt.subplot(1, 2, 2)
means = [(x + y) / 2 for (x, y) in zip(organoID_areas, manual_areas)]
differences = [(x - y) for (x, y) in zip(organoID_areas, manual_areas)]
plt.plot(means, differences, 'o')

mean = np.mean(differences)
std = np.std(differences)
labels = [(mean, "Mean=%.2f" % mean, "b"),
          (mean - std * 1.96, "-1.96\u03C3=%.2f" % (mean - std * 1.96), "r"),
          (mean + std * 1.96, "+1.96\u03C3=%.2f" % (mean + std * 1.96), "r")]
plt.axhline(y=mean, color="b", linestyle="solid")
plt.axhline(y=mean + std * 1.96, color="r", linestyle="dashed")
plt.axhline(y=mean - std * 1.96, color="r", linestyle="dashed")
plt.xlabel(r"Average of OrganoID and manual area ($\mu m^2$)")
plt.ylabel("Difference between OrganoID and manual area")
plt.title("Bland-Altman plot of OrganoID and manual organoid area")
plt.ylim([min(differences) - 2, -min(differences) + 2])
for (y, text, color) in labels:
    plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right', color=color)
plt.show()
