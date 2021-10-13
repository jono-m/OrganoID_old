from pathlib import Path
from backend.Segmenter import Segmenter
from backend.ImageManager import LoadImages, LabelToRGB, ShowImage
from backend.Label import Label
from backend.PostProcessing import PostProcess
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.measure import regionprops
import scipy.ndimage as ndimage
from sklearn.metrics import auc
from backend.Segmenter import Segmenter
from backend.ImageManager import LoadImages, ContrastOp
import matplotlib.pyplot as plt


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


imagesPath = Path(r"C:\Users\jonoj\Documents\ML\TestingData\ACCData\images")
segmentationsPath = Path(r"C:\Users\jonoj\Documents\ML\TestingData\ACCData\segmentations")
modelPath = Path(r"assets\model.tflite")

images = LoadImages(imagesPath, (512, 512), mode="L")
segmentations = LoadImages(segmentationsPath, (512, 512), mode="1")

segmenter = Segmenter(modelPath)

confusionMatrices = []
for image, segmentation in zip(images, segmentations):
    print("Segmenting %s" % image.path.name)
    predicted = image.DoOperation(segmenter.Segment)
    cmats = [confusionMatrix(segmentationFrame, predictedFrame > 0.5) for
             segmentationFrame, predictedFrame in zip(segmentation.frames, predicted.frames)]
    confusionMatrices += cmats

ious = [IOU(list(cmat.flatten())) for cmat in confusionMatrices]

ci = 2.262 * np.std(ious)/np.sqrt(len(ious))
mean = np.mean(ious)
print("Mean IOU: %.4f (95%% CI %.4f-%.4f)" % (mean, mean-ci, mean+ci))
print("Variance: " + str(np.var(ious)))