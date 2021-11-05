from pathlib import Path
import numpy as np
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


predictionsPath = Path(r"Z:\ML_Organoid\Paper\Data\Testing Predictions\*_threshold*")
groundTruthPath = Path(r"Z:\ML_Organoid\Paper\Data\Testing Ground Truth")
modelPath = Path(r"assets\model.tflite")

predictions = LoadImages(predictionsPath, (512, 512), mode="1")
groundTruths = LoadImages(groundTruthPath, (512, 512), mode="1")
groundTruths = list(groundTruths)
confusionMatrices = []
filenames = [gt.path for gt in groundTruths]
for prediction, groundTruth in zip(predictions, groundTruths):
    cmats = [confusionMatrix(segmentationFrame, predictedFrame) for
             segmentationFrame, predictedFrame in zip(groundTruth.frames, prediction.frames)]
    confusionMatrices += cmats

ious = [IOU(list(cmat.flatten())) for cmat in confusionMatrices]

csvFile = open(r"C:\Users\jonoj\Documents\ML\ious.csv", "w+")
csvFile.write("Filename, IOU\n")
for filename, iou in zip(filenames, ious):
    csvFile.write(filename.name + ", " + str(iou) + "\n")

mean = np.mean(ious)
print("Mean IOU: %.4f (95%% CI %.4f-%.4f)" % (mean, mean-ci, mean+ci))
print("Variance: " + str(np.var(ious)))