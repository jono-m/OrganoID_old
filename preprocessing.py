from PIL import Image
from pathlib import Path
import singleLineLogging


def PreprocessImages(jobID: str, trainingImagesPath: str, trainingSegmentationsPath: str):
    print("-----------------------")
    print("Preprocessing training data...")
    print("\tImages directory: " + trainingImagesPath)
    print("\tSegmentations directory: " + trainingSegmentationsPath)

    imagesOutPath = Path(trainingImagesPath).resolve() / ('OrganoID_pp_' + jobID)
    segmentationsOutPath = Path(trainingSegmentationsPath).resolve() / ('OrganoID_pp_' + jobID)

    imagesOutPath.mkdir(exist_ok=True)
    segmentationsOutPath.mkdir(exist_ok=True)

    imagePaths = [file for file in Path(trainingImagesPath).iterdir() if file.is_file()]
    segmentationPaths = [file for file in Path(trainingSegmentationsPath).iterdir() if file.is_file()]

    numImages = len(list(imagePaths))

    imageNumber = 1
    for imagePath in imagePaths:
        singleLineLogging.DoLog("Processing image " + str(imageNumber) + "/" + str(numImages) + "...")
        filename = imagePath.name
        outputPath = imagesOutPath / filename
        image = Image.open(str(imagePath.resolve()))
        image.convert(mode='L')
        image.save(str(outputPath.resolve()))
        imageNumber += 1

    singleLineLogging.ClearLog()

    print("\tDone!")

    numSegmentations = len(list(segmentationPaths))
    segmentationNumber = 1
    for segmentationPath in segmentationPaths:
        singleLineLogging.DoLog("Processing segmentation " + str(segmentationNumber) + "/" + str(numSegmentations) + "...")
        filename = segmentationPath.name
        outputPath = segmentationsOutPath / filename
        image = Image.open(str(segmentationPath.resolve()))
        image.convert(mode='1')
        image.save(str(outputPath.resolve()))
        segmentationNumber += 1

    singleLineLogging.ClearLog()

    print("\tDone!")

    results = str(imagesOutPath.resolve()), str(segmentationsOutPath.resolve())

    print("Preprocessed images saved to '" + results[0])
    print("Preprocessed segmentations saved to '" + results[1])
    print("-----------------------")

    return results
