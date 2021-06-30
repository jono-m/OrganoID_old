import numpy as np
import tensorflow as tf
from PIL import Image


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, imagePaths, segmentationPaths, imageSize, batchSize):
        self.batchSize = batchSize
        self.imagePaths = np.array(imagePaths)
        self.imageSize = imageSize
        self.segmentationPaths = np.array(segmentationPaths)
        self.on_epoch_end()

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
        images = [Image.open(imagePath) for imagePath in imagePaths]
        for imageIndex, image in enumerate(images):
            if image.mode == 'I' or image.mode == 'I;16':
                image = image.point(lambda x: x * (1 / 255))
            images[imageIndex] = np.array(image.convert(mode="L").resize(self.imageSize))
        images = np.expand_dims(np.moveaxis(np.stack(images, axis=-1), -1, 0), -1)
        return images

    def LoadSegmentations(self, segmentationPaths):
        segmentations = [Image.open(segmentationPath) for segmentationPath in segmentationPaths]
        for segmentationIndex, segmentation in enumerate(segmentations):
            segmentations[segmentationIndex] = np.array(segmentation.convert(mode="1").resize(self.imageSize))

        segmentations = np.moveaxis(np.stack(segmentations, axis=-1), -1, 0).astype(int)
        return segmentations
