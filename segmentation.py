from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np


def DoSegmentation(imagesPath: Path, outputPath: Path, modelPath: Path, useGPU):
    if not useGPU:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("-----------------------")
    print("Running segmentation pipeline...")
    print("\tImages directory: " + str(imagesPath))
    print("\tModel path: " + str(modelPath))
    print("\tOutput directory: " + str(outputPath))

    print("\tLoading model.")
    model = tf.keras.models.load_model(str(modelPath.absolute()), compile=False)
    model.compile(loss=tf.keras.losses.binary_crossentropy)
    print("\tDone.")

    imagePaths = [imagePath for imagePath in imagesPath.iterdir() if imagePath.is_file()]

    print("\tDone.")

    outputPath.mkdir(parents=True, exist_ok=True)

    for imageIndex, imagePath in enumerate(imagePaths):
        print("\tSegmenting image " + str(imageIndex + 1) + "/" + str(len(imagePaths)))

        segmented = SegmentImage(imagePath, model)
        (unique, counts) = np.unique(segmented, return_counts=True)
        frequencies = np.asarray((unique, counts))
        print("\t\tFrequencies: " + str(frequencies))
        outputFilename = imagePaths[imageIndex].stem + "_seg" + imagePaths[imageIndex].suffix
        finalOutput = outputPath / outputFilename
        Image.fromarray(segmented > 0.5).convert(mode="1").save(finalOutput)


def SegmentImage(imagePath: Path, model):
    image = Image.open(imagePath)
    if image.mode == 'I':
        image = image.point(lambda x: x * (1 / 255))
    inputShape = model.layers[0].input_shape[0]
    imagePrepared = np.reshape(np.array(image.resize(inputShape[1:3]).convert(mode="L")), [1] + list(inputShape[1:]))
    segmented = model.predict(imagePrepared)[0, :, :, 0]
    return segmented
