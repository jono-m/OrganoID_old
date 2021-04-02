import tensorflow as tf
from PIL import Image
from pathlib import Path
import numpy as np


def TestSegmentImage(imagePath: Path, modelPath: Path):
    model = tf.keras.models.load_model(modelPath, compile=False)
    imageSize = tuple(model.inputs[0].shape[1:3])

    rawImage = Image.open(imagePath)
    convertedImage = np.array(rawImage.resize(imageSize).convert(mode="RGB"))
    preparedImage = np.expand_dims(convertedImage, axis=0)

    predictedSegmentationProbabilities = model.predict(preparedImage)[0, :, :, 0]

    segmentedImage = np.greater(predictedSegmentationProbabilities, 0.5)

    rgb = np.stack([np.zeros(imageSize, np.uint8), segmentedImage.astype(np.uint8), np.zeros(imageSize, np.uint8)],
                   2) * 255

    transparencyMask = 255 - segmentedImage.astype(np.uint8) * 128

    merged = Image.composite(Image.fromarray(convertedImage), Image.fromarray(rgb), Image.fromarray(transparencyMask))
    merged.show()


TestSegmentImage(
    imagePath=Path(
        r"C:\Users\jonoj\Documents\ML\OrganoID_augment_2021_04_01_19_09_21\images\0d12b67f-c9fd-4013-8735-7d4e1023e4de.png"),
    modelPath=Path(r"C:\Users\jonoj\Documents\ML\Model\trainedModel.h5"))
