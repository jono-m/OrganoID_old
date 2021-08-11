import tensorflow as tf
import numpy as np
from backend.ImageManager import LoadImages, ContrastOp
from pathlib import Path


class ModelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, imagePaths: Path, segmentationPaths, imageSize, batchSize, contrast=True):
        self.batchSize = batchSize
        self.imagePaths = np.array([path for path in imagePaths.iterdir() if path.is_file()])
        self.imageSize = imageSize
        self.segmentationPaths = np.array([path for path in segmentationPaths.iterdir() if path.is_file()])
        self.on_epoch_end()
        self.contrast = contrast

    def __len__(self):
        return int(np.ceil(len(self.imagePaths) / self.batchSize))

    def __getitem__(self, batchNumber):
        batchStart = batchNumber * self.batchSize
        batchEnd = (batchNumber + 1) * self.batchSize
        imagePaths = self.imagePaths[batchStart:batchEnd]
        segmentationPaths = self.segmentationPaths[batchStart:batchEnd]

        images = self.LoadImages(imagePaths)
        segmentations = self.LoadSegmentations(segmentationPaths)

        return images, segmentations

    def on_epoch_end(self):
        indexes = np.random.permutation(len(self.imagePaths))
        self.imagePaths = self.imagePaths[indexes]
        self.segmentationPaths = self.segmentationPaths[indexes]

    def LoadImages(self, imagePaths):
        imageData = np.zeros([len(imagePaths), self.imageSize[0], self.imageSize[1], 1], dtype=np.uint8)

        for imageIndex in range(len(imagePaths)):
            image = next(LoadImages(imagePaths[imageIndex], size=self.imageSize, mode="L"))
            if self.contrast:
                image = image.DoOperation(ContrastOp)
            imageData[imageIndex, :, :, 0] = image.frames[0]
        return imageData

    def LoadSegmentations(self, segmentationPaths):
        segmentationData = np.zeros([len(segmentationPaths), self.imageSize[0], self.imageSize[1], 1], dtype=np.uint8)

        for segmentationIndex in range(len(segmentationPaths)):
            segmentationData[segmentationIndex, :, :, 0] = next(LoadImages(segmentationPaths[segmentationIndex],
                                                                           size=self.imageSize, mode="1")).frames[0]

        return segmentationData
