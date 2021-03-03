from PIL import Image
from pathlib import Path
import singleLineLogging


def PreprocessImages(jobID: str, trainingImagesPath: Path, trainingSegmentationsPath: Path):
    print("-----------------------")
    print("Preprocessing training data...")
    print("\tImages directory: " + str(trainingImagesPath))
    print("\tSegmentations directory: " + str(trainingSegmentationsPath))

    imagesOutPath = trainingImagesPath / ('OrganoID_pp_' + jobID)
    segmentationsOutPath = trainingSegmentationsPath / ('OrganoID_pp_' + jobID)

    imagesOutPath.mkdir(exist_ok=True)
    segmentationsOutPath.mkdir(exist_ok=True)

    imagePaths = [file for file in trainingImagesPath.iterdir() if file.is_file()]
    segmentationPaths = [file for file in trainingSegmentationsPath.iterdir() if file.is_file()]

    numImages = len(list(imagePaths))

    imageNumber = 1
    for imagePath in imagePaths:
        singleLineLogging.DoLog("Processing image " + str(imageNumber) + "/" + str(numImages) + "...")
        filename = imagePath.name
        outputPath = imagesOutPath / filename
        image = Image.open(imagePath)
        image.convert(mode='L')
        image.save(str(outputPath))
        imageNumber += 1

    singleLineLogging.ClearLog()

    print("\tDone!")

    numSegmentations = len(list(segmentationPaths))
    segmentationNumber = 1
    for segmentationPath in segmentationPaths:
        singleLineLogging.DoLog(
            "Processing segmentation " + str(segmentationNumber) + "/" + str(numSegmentations) + "...")
        filename = segmentationPath.name
        outputPath = segmentationsOutPath / filename
        image = Image.open(segmentationPath.resolve())
        image.convert(mode='1')
        image.save(outputPath.resolve())
        segmentationNumber += 1

    singleLineLogging.ClearLog()

    print("\tDone!")

    results = imagesOutPath, segmentationsOutPath

    print("Preprocessed images saved to '" + str(results[0]))
    print("Preprocessed segmentations saved to '" + str(results[1]))
    print("-----------------------")

    return results
