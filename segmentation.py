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
    model = tf.keras.models.load_model(modelPath)
    print("\tDone.")

    print("\tLoading images.")
    imagePaths = [imagePath for imagePath in imagesPath.iterdir() if imagePath.is_file()]
    images = [Image.open(imagePath) for imagePath in imagePaths]

    print("\tDone.")
    imageSize = images[0].size

    outputPath.mkdir(parents=True, exist_ok=True)

    for imageIndex, image in enumerate(images):
        print("\tConverting image " + str(imageIndex + 1) + "/" + str(len(images)))
        if image.mode == 'I':
            image = image.point(lambda x: x * (1 / 255))
        imagePrepared = np.expand_dims(np.array(image.resize(imageSize).convert(mode="L")), axis=0)
        print(imagePrepared.shape)
        print("\tSegmenting image " + str(imageIndex + 1) + "/" + str(len(images)))
        segmented = model.predict(imagePrepared)[0, :, :, 0]
        (unique, counts) = np.unique(segmented, return_counts=True)
        frequencies = np.asarray((unique, counts))
        print("\t\tFrequencies: " + str(frequencies))
        print("\tSaving image " + str(imageIndex + 1) + "/" + str(len(images)))
        outputFilename = imagePaths[imageIndex].stem + "_seg" + imagePaths[imageIndex].suffix
        finalOutput = outputPath / outputFilename
        Image.fromarray(segmented).save(finalOutput)
