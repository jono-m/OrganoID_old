import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
from training import tf
from PIL import Image, ImageOps
import numpy as np


def LoadH5Model(modelPath, weightsPath):
    file = open(modelPath)
    json = file.read()
    loadedModel = tf.keras.models.model_from_json(json)
    file.close()
    loadedModel.load_weights(weightsPath)
    return loadedModel


def LoadModel(modelPath):
    loadedModel = tf.keras.models.load_model(modelPath)
    return loadedModel


def SegmentImage(imagePath: Path, model, mode):
    image = Image.open(imagePath)
    if image.mode == 'I':
        image = image.point(lambda x: x * (1 / 255))
    inputShape = model.layers[0].input_shape[0]
    imagePrepared = np.reshape(np.array(image.resize(inputShape[1:3]).convert(mode=mode)), [1] + list(inputShape[1:]))
    segmented = model.predict(imagePrepared)
    print(segmented.shape)

    return imagePrepared, segmented


def ShowImage(prepared, segmented):
    Image.fromarray(segmented[0, :, :, 0]).show()


model = LoadModel("C:/Users/jonoj/Documents/ML/OrganoID_train_2021_06_22_11_53_51/trainedModel")
(prepared, segmented) = SegmentImage(Path("C:/Users/jonoj/Documents/ML/X/51.png"),
                                     model, "RGB")
(unique, counts) = np.unique(segmented, return_counts=True)
frequencies = np.asarray((unique, counts)).T

ShowImage(prepared, segmented)
print(frequencies)
input()
