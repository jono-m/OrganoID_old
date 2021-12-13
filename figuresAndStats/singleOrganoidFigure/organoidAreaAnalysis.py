import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
from backend.Detector import Detector
from backend.ImageManager import LoadImages, LabelToRGB, ShowImage
from backend.Label import Label
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndimage
from util import Printer

# Set this to TRUE to display the matched organoids between OrganoID- and manual-labeled
# images.
displayComparison = False


def OverlapCost(coordinatesA, bCoordinates):
    overlaps = []
    for coordinatesB in bCoordinates:
        overlap = np.count_nonzero((coordinatesA[:, None] == coordinatesB).all(-1).any(-1))
        if overlap == 0:
            overlap = 10000
        else:
            overlap = 1 / overlap
        overlaps.append(overlap)
    return np.asarray(overlaps)


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

file = open(r"figuresAndStats\singleOrganoidFigure\data\matched_areas.csv", "w+")
file.write("Image, Manual label, Manual area, OrganoID label, OrganoID area\n")

for (image, segmentation) in zip(images, segmentations):
    print("Analyzing image %s " % image.path.name)
    detected = detector.Detect(image.frames[0])
    organoID_labeled = Label(detected, 200, False)

    manual_labeled, _ = ndimage.label(segmentation.frames[0])
    manual_labeled = remove_small_objects(manual_labeled, 200)

    organoIDOrganoids = regionprops(organoID_labeled)
    manualOrganoids = regionprops(manual_labeled)

    # Now we match organoids in the manual and OrganoID labeled images by POSITION only
    # (to compare area). Use Hungarian assignment algorithm with cost matrix of overlap amount.
    numPredicted = len(organoIDOrganoids)
    numActual = len(manualOrganoids)
    fullSize = max(numPredicted, numActual)
    costMatrix = np.zeros([fullSize, fullSize])
    coordinatesOID = [oid.coords for oid in organoIDOrganoids]
    coordinatesManual = [manual.coords for manual in manualOrganoids]

    n = len(coordinatesOID)
    for i, oidCoord in enumerate(coordinatesOID):
        Printer.printRep("%d/%d" % (i + 1, n))
        costs = OverlapCost(oidCoord, coordinatesManual) * 100
        costMatrix[i, 0:len(coordinatesManual)] = costs
    predictedIndices, manualIndices = linear_sum_assignment(costMatrix)
    Printer.printRep()

    # Gather areas for the position-matched organoids
    pairCount = min(numPredicted, numActual)
    pairs = [pair for (i, pair) in enumerate(zip(predictedIndices, manualIndices)) if
             pair[0] < pairCount and pair[1] < pairCount]
    labelPairs = [(organoIDOrganoids[predictedIndex].label, manualOrganoids[manualIndex].label)
                  for (predictedIndex, manualIndex) in pairs]
    areaPairs = [
        (organoIDOrganoids[predictedIndex].area * 6.8644, manualOrganoids[manualIndex].area * 6.8644)
        for (predictedIndex, manualIndex) in pairs]

    [file.write("%s, %d, %f, %d, %f\n" % (image.path.name, labels[0], areas[0], labels[1], areas[1])) for labels, areas
     in
     zip(labelPairs, areaPairs)]
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

file.close()
