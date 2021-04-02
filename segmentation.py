from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np


def DoSegmentation(imagesPath: Path, outputPath: Path, modelPath: Path):
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

    outputPath.mkdir(exist_ok=True)

    for imageIndex, image in enumerate(images):
        print("\tConverting image " + str(imageIndex + 1) + "/" + str(len(images)))
        if image.mode == 'I':
            image = image.point(lambda x: x * (1 / 255))
        imagePrepared = np.expand_dims(np.array(image.resize(imageSize).convert(mode="RGB")), axis=0)
        print(imagePrepared.shape)
        print("\tSegmenting image " + str(imageIndex + 1) + "/" + str(len(images)))
        segmented = model.predict(imagePrepared)[0, :, :, 0].astype(bool)
        print(segmented.shape)
        print("\tSaving image " + str(imageIndex + 1) + "/" + str(len(images)))
        outputFilename = imagePaths[imageIndex].stem + "_seg" + imagePaths[imageIndex].suffix
        finalOutput = outputPath / outputFilename
        print(segmented.dtype)
        Image.fromarray(segmented).save(finalOutput)
