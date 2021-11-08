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


imagesPath = Path(r"dataset\testing\images\PDAC*")
segmentationsPath = Path(r"dataset\testing\segmentations\PDAC*")
modelPath = Path(r"model\model.tflite")

images = LoadImages(imagesPath, (512, 512), mode="L")
segmentations = LoadImages(segmentationsPath, (512, 512), mode="1")

detector = Detector(modelPath)

confusionMatrices = []
for (image, segmentation) in zip(images, segmentations):
    print("Analyzing image %s " % image.path.name)
    detectedFrames = [detector.Detect(frame) >= 0.5 for frame in image.frames]
    cmats = [confusionMatrix(segmentationFrame, predictedFrame) for
             segmentationFrame, predictedFrame in zip(segmentation.frames, detectedFrames)]
    confusionMatrices += cmats

ious = [IOU(list(cmat.flatten())) for cmat in confusionMatrices]

mean = np.mean(ious)
std = np.std(ious)
print("Mean IOU: %.4f (STD=%.4f)" % (mean, std))
