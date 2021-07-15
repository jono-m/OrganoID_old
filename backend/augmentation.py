import Augmentor
from pathlib import Path
import re
import typing
import shutil

from sklearn.model_selection import train_test_split


def Augment(outputPath: Path, imagesPath: Path, segmentationsPath: Path, testSplit: float,
            size: typing.Tuple[int, int], augmentCount: int):
    imagePaths = [imagePath for imagePath in sorted(imagesPath.iterdir()) if imagePath.is_file()]
    segmentationPaths = [segmentationPath for segmentationPath in sorted(segmentationsPath.iterdir())
                         if segmentationPath.is_file()]

    if testSplit == 0:
        rawImagesPath = outputPath / "raw" / "images"
        rawSegmentationsPath = outputPath / "raw" / "segmentations"
        Copy(imagePaths, rawImagesPath)
        Copy(segmentationPaths, rawSegmentationsPath)

        print("-----------------------")
        print("Augmenting data...")
        RunPipeline(rawImagesPath, rawSegmentationsPath, outputPath / "augmented", size, augmentCount)
        print("-----------------------")
    else:
        print("\tSplitting training and testing datasets (" + str(testSplit * 100) + "% for testing)...")
        rawTrainingImagesPath = outputPath / "raw" / "training" / "images"
        rawTrainingSegmentationsPath = outputPath / "raw" / "training" / "segmentations"
        rawTestingImagesPath = outputPath / "raw" / "testing" / "images"
        rawTestingSegmentationsPath = outputPath / "raw" / "testing" / "segmentations"
        trainingImagePaths, testingImagePaths, trainingSegmentationPaths, testingSegmentationPaths = train_test_split(
            imagePaths,
            segmentationPaths,
            test_size=testSplit)

        Copy(trainingImagePaths, rawTrainingImagesPath)
        Copy(trainingSegmentationPaths, rawTrainingSegmentationsPath)
        Copy(testingImagePaths, rawTestingImagesPath)
        Copy(testingSegmentationPaths, rawTestingSegmentationsPath)

        print("-----------------------")
        print("Augmenting data...")
        RunPipeline(rawTrainingImagesPath, rawTrainingSegmentationsPath, outputPath / "training", size, augmentCount)
        RunPipeline(rawTestingImagesPath, rawTestingSegmentationsPath, outputPath / "testing", size,
                    int(augmentCount * testSplit))
        print("-----------------------")


def Copy(paths: typing.List[Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        newPath = output / path.name
        shutil.copy(path, newPath)


def RunPipeline(imagesPath: Path, segmentationPath: Path, outputPath: Path, size, augmentCount):
    outputPath.mkdir(parents=True, exist_ok=True)

    augmentor = Augmentor.Pipeline(source_directory=imagesPath,
                                   output_directory=outputPath)
    augmentor.set_save_format("auto")
    augmentor.ground_truth(segmentationPath)

    augmentor.resize(1, size[0], size[1])
    augmentor.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    augmentor.flip_left_right(probability=0.5)
    augmentor.flip_top_bottom(probability=0.5)
    augmentor.zoom_random(probability=0.5, percentage_area=0.7)
    augmentor.shear(probability=1, max_shear_left=20, max_shear_right=20)
    augmentor.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=3)
    augmentor.skew(probability=0.5, magnitude=0.3)
    augmentor.resize(1, size[0], size[1])

    augmentor.sample(augmentCount)

    outputImagesPath = outputPath / "images"
    outputImagesPath.mkdir(parents=True, exist_ok=True)
    outputSegmentationsPath = outputPath / "segmentations"
    outputSegmentationsPath.mkdir(parents=True, exist_ok=True)

    results = list(outputPath.glob("*.*"))
    trainingSegmentationFiles = list(outputPath.glob("_groundtruth*"))
    trainingImageFiles = [x for x in results if x not in trainingSegmentationFiles]

    for trainingSegmentationFile in trainingSegmentationFiles:
        newFilename = re.sub(".*_", "", trainingSegmentationFile.name)
        trainingSegmentationFile.rename(outputSegmentationsPath / newFilename)

    for trainingImageFile in trainingImageFiles:
        newFilename = re.sub(".*_", "", trainingImageFile.name)
        trainingImageFile.rename(outputImagesPath / newFilename)
