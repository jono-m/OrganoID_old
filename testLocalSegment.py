import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from PIL import Image
from pathlib import Path
import numpy as np


def TestSegmentImage(imagePath: Path, modelPath: Path):
    model = tf.keras.models.load_model(modelPath)
    imageSize = tuple(model.inputs[0].shape[1:3])

    rawImage = Image.open(imagePath)
    convertedImage = np.array(rawImage.resize(imageSize).convert(mode="RGB"))
    preparedImage = np.expand_dims(convertedImage, axis=0)

    predictedSegmentationProbabilities = model.predict(preparedImage)[0, :, :, 0]

    return predictedSegmentationProbabilities


image = TestSegmentImage(
    imagePath=Path(
        r"C:\Users\jonoj\Documents\ML\X\51.png"),
    modelPath=Path(r"C:\Users\jonoj\Documents\ML\OrganoID_train_2021_06_22_11_53_51"))

print(np.unique(image))
