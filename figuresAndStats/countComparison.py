from pathlib import Path
from backend.Segmenter import Segmenter
from backend.ImageManager import LoadImages, LabelToRGB
from backend.Label import Label
from backend.PostProcessing import PostProcess
import numpy as np
from skimage.measure import regionprops
import scipy.ndimage as ndimage


def MatchLabels(a: np.ndarray, b: np.ndarray, aLabels, bLabels, aIsMaster):
    if aIsMaster:
        newB = np.zeros_like(b)
        for (aLabel, bLabel) in zip(aLabels, bLabels):
            newB[b == bLabel] = aLabel
        b = newB
    else:
        newA = np.zeros_like(a)
        for (aLabel, bLabel) in zip(aLabels, bLabels):
            newA[a == aLabel] = bLabel
        a = newA
    return a, b


modelPath = Path(r"assets\model.tflite")
imagesPath = Path(r"C:\Users\jonoj\Documents\ML\TestingData\images")
segmentationsPath = Path(r"C:\Users\jonoj\Documents\ML\TestingData\segmentations")
outputPath = Path(r"C:\Users\jonoj\Documents\ML\SingleComparison\counts.csv")

segmenter = Segmenter(modelPath)

images = LoadImages(imagesPath, [512, 512], mode="L")
segmentations = LoadImages(segmentationsPath, [512, 512], mode="1")

organoidAreas = []
for (image, segmentation) in zip(images, segmentations):
    predicted = segmenter.Segment(image.frames[0])
    predictedLabeled = PostProcess(Label(predicted, 0.5))

    manualLabeled, _ = ndimage.label(segmentation.frames[0])

    predictedOrganoids = regionprops(predictedLabeled)
    actualOrganoids = regionprops(manualLabeled)

    numPredicted = len(predictedOrganoids)
    numManual = len(actualOrganoids)

    organoidAreas.append((image.path.name, [organoid.area for organoid in predictedOrganoids],
                          [organoid.area for organoid in actualOrganoids]))

    print("%s, %d, %d\n" % (image.path.name, numPredicted, numManual))

outputPath.parent.mkdir(exist_ok=True, parents=True)
dumpFile = open(outputPath, "w+")
for organoidArea in organoidAreas:
    name, predictedAreas, manualAreas = organoidArea
    dumpFile.write(name + "Predicted, " + ", ".join([str(x) for x in predictedAreas]) + "\n")
    dumpFile.write(name + "Manual, " + ", ".join([str(x) for x in manualAreas]) + "\n")
