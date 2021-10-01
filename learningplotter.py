from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.losses import BinaryCrossentropy, binary_crossentropy
from backend.ImageManager import LoadImages, sort_paths_nicely
from backend.Segmenter import Segmenter

modelsPath = Path(r"assets\epochs")
imagesPath = Path(r"C:\Users\jonoj\Documents\ML\AugustPreparedData\training\images")
segmentationsPath = Path(r"C:\Users\jonoj\Documents\ML\AugustPreparedData\training\segmentations")
outputPath = Path(r"C:\Users\jonoj\Documents\ML\EpochData\trainingLoss.csv")

modelPaths = list(modelsPath.iterdir())
sort_paths_nicely(modelPaths)


def ComputeLoss(modelPath):
    segmenter = Segmenter(modelPath)

    bceComputer = BinaryCrossentropy()

    images = LoadImages(imagesPath, (512, 512), mode="L")
    segmentations = LoadImages(segmentationsPath, (512, 512), mode="1")

    epochLosses = []
    for (image, segmentation) in zip(images, segmentations):
        predicted = segmenter.Segment(image.frames[0])
        loss = bceComputer(segmentation.frames[0].astype(np.float32), predicted).numpy()
        print("\t" + image.path.name + " - " + str(loss))
        epochLosses.append(loss)

    mean = np.mean(epochLosses)
    return mean


def ComputeLossInvert(modelPaths):
    segmenters = [Segmenter(modelPath) for modelPath in modelPaths]

    bceComputer = BinaryCrossentropy()

    images = LoadImages(imagesPath, (512, 512), mode="L")
    segmentations = LoadImages(segmentationsPath, (512, 512), mode="1")

    epochLosses = [[] for _ in range(len(segmenters))]
    for (image, segmentation) in zip(images, segmentations):
        print(image.path.name + ":\n")
        for epoch, segmenter in enumerate(segmenters):
            predicted = segmenter.Segment(image.frames[0])
            loss = bceComputer(segmentation.frames[0].astype(np.float32), predicted).numpy()
            print("\t" + str(epoch) + ": " + str(loss))
            epochLosses[epoch].append(loss)

    losses = [np.mean(epochLoss) for epochLoss in epochLosses]
    return losses


def ComputeLossBatch(modelPath):
    segmenter = Segmenter(modelPath)

    images = LoadImages(imagesPath, (512, 512), mode="L")
    segmentations = LoadImages(segmentationsPath, (512, 512), mode="1")
    images = [image.frames[0] for image in images]
    segmentations = np.stack([segmentation.frames[0] for segmentation in segmentations])

    predicted = segmenter.SegmentMultiple(images)
    loss = binary_crossentropy(segmentations.astype(np.float32), predicted)

    mean = np.mean(loss)
    return mean


#
# losses = []
#
# for epochModelPath in modelPaths:
#     losses.append(ComputeLoss(epochModelPath))

losses = ComputeLossInvert(modelPaths)

outputPath.parent.mkdir(parents=True, exist_ok=True)
file = open(outputPath, "w")
file.write(",".join([str(loss) for loss in losses]))
file.close()
