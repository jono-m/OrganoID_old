from pathlib import Path
from PIL import Image


def DoSegmentation(imagesPath: Path, outputPath: Path, modelPath: Path, useGPU, threshold):
    if not useGPU:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from segment_lite import SmartInterpreter

    print("-----------------------")
    print("Running segmentation pipeline...")
    print("\tImages directory: " + str(imagesPath))
    print("\tModel path: " + str(modelPath))
    print("\tOutput directory: " + str(outputPath))

    print("\tLoading model.")
    interpreter = SmartInterpreter(modelPath)
    print("\tDone.")

    imagePaths = [imagePath for imagePath in imagesPath.iterdir() if imagePath.is_file()]

    print("\tDone.")

    outputPath.mkdir(parents=True, exist_ok=True)

    for imageIndex, imagePath in enumerate(imagePaths):
        print("\tSegmenting image " + str(imageIndex + 1) + "/" + str(len(imagePaths)))

        image = Image.open(imagePath)
        segmentedImage = Image.fromarray(interpreter.Predict(PrepareImage(image, interpreter.inputShape)) > threshold)

        # Save segmentation
        segmentedFilename = outputPath / ("seg_" + imagePaths[imageIndex].stem + imagePaths[imageIndex].suffix)
        segmentedImage.save(segmentedFilename)

        # Save blended
        blendedFilename = outputPath / ("blend_ " + imagePaths[imageIndex].stem + imagePaths[imageIndex].suffix)
        blendedImage = BlendImage(image, segmentedImage)
        blendedImage.save(blendedFilename)


def BlendImage(image: Image, segmentation: Image):
    if image.mode == 'I' or image.mode == 'I;16':
        image = image.point(lambda x: x * (1 / 255))
    imageRGB = image.convert(mode="RGB").resize(segmentation.size)
    overlayRGB = Image.new(mode="RGB", size=imageRGB.size, color=(0, 255, 0))
    segmentationMask = segmentation.convert(mode="L").point(lambda pixel: int(pixel) / 2)
    composite = Image.composite(overlayRGB, imageRGB, segmentationMask)
    return composite


def PrepareImage(image: Image, shape):
    if image.mode == 'I' or image.mode == 'I;16':
        image = image.point(lambda x: x * (1 / 255))
    prepared = image.resize(shape).convert(mode="L")
    return prepared


def SegmentImage(image: Image, model):
    segmented = model.predict(PrepareImage(image, model))[0, :, :, 0]
    return segmented
