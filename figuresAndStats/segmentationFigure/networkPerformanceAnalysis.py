from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))
import numpy as np
from backend.Detector import Detector
from backend.ImageManager import LoadImages


def IOU(cmat):
    (TP, FP, FN, TN) = cmat
    return TP / (FP + FN + TP)


def TPR(cmat):
    (TP, FP, FN, TN) = cmat
    return TP / (TP + FN)


def FPR(cmat):
    (TP, FP, FN, TN) = cmat
    return FP / (FP + TN)


def confusionMatrix(true: np.ndarray, predicted: np.ndarray):
    true = true.astype(dtype=bool)
    predicted = predicted.astype(dtype=bool)
    TP = np.count_nonzero(np.bitwise_and(true == predicted, true))
    FP = np.count_nonzero(np.bitwise_and(true != predicted, np.bitwise_not(true)))
    FN = np.count_nonzero(np.bitwise_and(true != predicted, true))
    TN = np.count_nonzero(np.bitwise_and(true == predicted, np.bitwise_not(true)))
    return np.asarray([[TP, FP], [FN, TN]])


datasets = ["PDAC", "ACC", "C", "Lung"]
names = ["PDAC", "ACC", "Colon", "Lung"]

modelPath = Path(r"model\model.tflite")
detector = Detector(modelPath)

ious = []

for dataset in datasets:
    imagesPath = Path(r"dataset\testing\images\%s*" % dataset)
    segmentationsPath = Path(r"dataset\testing\segmentations\%s*" % dataset)
    images = LoadImages(imagesPath, (512, 512), mode="L")
    segmentations = LoadImages(segmentationsPath, (512, 512), mode="1")
    confusionMatrices = []
    for (image, segmentation) in zip(images, segmentations):
        print("Analyzing image %s " % image.path.name)
        detectedFrames = [detector.Detect(frame) >= 0.5 for frame in image.frames]
        cmats = [confusionMatrix(segmentationFrame, predictedFrame) for
                 segmentationFrame, predictedFrame in zip(segmentation.frames, detectedFrames)]
        confusionMatrices += cmats

    ious.append([IOU(list(cmat.flatten())) for cmat in confusionMatrices])

dataFile = open(r"figuresAndStats\segmentationFigure\data\ious.csv", "w+")
for name, iou in zip(names, ious):
    dataFile.write(name + ", " + ", ".join([str(x) for x in iou]) + "\n")
dataFile.close()
