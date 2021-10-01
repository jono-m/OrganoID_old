from pathlib import Path
from backend.Segmenter import Segmenter
from backend.ImageManager import LoadImages, LabelToRGB, ShowImage
from backend.Label import Label
from backend.PostProcessing import PostProcess
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.measure import regionprops
import scipy.ndimage as ndimage


def Relabel(a: np.ndarray, b: np.ndarray, labelPairs):
    newA = np.zeros_like(a)
    newB = np.zeros_like(b)

    for (i, (aLabel, bLabel)) in enumerate(labelPairs):
        newA[a == aLabel] = i
        newB[b == bLabel] = i
    return newA, newB


modelPath = Path(r"assets\model.tflite")
imagesPath = Path(r"C:\Users\jonoj\Documents\ML\TestingData\images")
segmentationsPath = Path(r"C:\Users\jonoj\Documents\ML\TestCountData")
outputPath = Path(r"C:\Users\jonoj\Documents\ML\SingleComparison\areas.csv")

segmenter = Segmenter(modelPath)

images = LoadImages(imagesPath, [512, 512], mode="L")
segmentations = LoadImages(segmentationsPath, [512, 512], mode="1")

organoidAreas = []
for (image, segmentation) in zip(images, segmentations):
    predicted = segmenter.Segment(image.frames[0])
    predictedLabeled = PostProcess(Label(predicted, 0.5))

    manualLabeled, _ = ndimage.label(segmentation.frames[0])

    predictedOrganoids = regionprops(predictedLabeled)
    manualOrganoids = regionprops(manualLabeled)

    numPredicted = len(predictedOrganoids)
    numActual = len(manualOrganoids)

    fullSize = max(numPredicted, numActual)
    costMatrix = np.zeros([fullSize, fullSize])
    for predictedIndex, predictedOrganoid in enumerate(predictedOrganoids):
        for manualIndex, manualOrganoid in enumerate(manualOrganoids):
            distance = np.sqrt((predictedOrganoid.centroid[0] - manualOrganoid.centroid[0]) ** 2 +
                               (predictedOrganoid.centroid[1] - manualOrganoid.centroid[1]) ** 2)
            costMatrix[predictedIndex, manualIndex] = distance

    predictedIndices, manualIndices = linear_sum_assignment(costMatrix)

    pairCount = min(numPredicted, numActual)
    pairs = [pair for (i, pair) in enumerate(zip(predictedIndices, manualIndices)) if pair[0] < pairCount and pair[1] < pairCount]
    labelPairs = [(predictedOrganoids[predictedIndex].label, manualOrganoids[manualIndex].label)
                  for (predictedIndex, manualIndex) in pairs]
    areaPairs = [(predictedOrganoids[predictedIndex].area, manualOrganoids[manualIndex].area)
                 for (predictedIndex, manualIndex) in pairs]

    newA, newB = Relabel(predictedLabeled, manualLabeled, labelPairs)

    organoidAreas.append((image.path.name, areaPairs))

    vis = False
    if vis:
        aImage = LabelToRGB(newA, True)
        bImage = LabelToRGB(newB, True)

        mergedA = np.concatenate([LabelToRGB(predictedLabeled, True), LabelToRGB(manualLabeled, True)], axis=1)
        mergedB = np.concatenate([aImage, bImage], axis=1)
        merged = np.concatenate([mergedA, mergedB])

        ShowImage(merged)
        input()
    print(image.path.name + ": " + str(pairCount))

outputPath.parent.mkdir(exist_ok=True, parents=True)
dumpFile = open(outputPath, "w+")
for organoidArea in organoidAreas:
    name, areaPairs = organoidArea
    predictedAreas = [pair[0] for pair in areaPairs]
    manualAreas = [pair[1] for pair in areaPairs]
    dumpFile.write(name + " Predicted, " + ", ".join([str(x) for x in predictedAreas]) + "\n")
    dumpFile.write(name + " Manual, " + ", ".join([str(x) for x in manualAreas]) + "\n")
