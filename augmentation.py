import Augmentor
from pathlib import Path
import re
import SettingsParser
import typing


def DoAugment(settings: SettingsParser):
    Augment(settings.OutputPath(),
            settings.ImagesPath(),
            settings.SegmentationsPath(),
            settings.GetSize(),
            settings.AugmentCount())


def Augment(outputPath: Path, trainingImagesPath: Path, trainingSegmentationsPath: Path, size: typing.Tuple[int, int],
            augmentCount: int):
    outputPath.mkdir(parents=True, exist_ok=True)

    print("-----------------------")
    print("Augmenting training data...")
    print("\tImages directory: " + str(trainingImagesPath))
    print("\tSegmentations directory: " + str(trainingSegmentationsPath))

    augmentor = Augmentor.Pipeline(source_directory=trainingImagesPath,
                                   output_directory=outputPath)
    augmentor.set_save_format("auto")
    augmentor.ground_truth(trainingSegmentationsPath)

    augmentor.resize(1, size[0], size[1])
    augmentor.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    augmentor.flip_left_right(probability=0.5)
    augmentor.flip_top_bottom(probability=0.5)
    augmentor.zoom_random(percentage_area=0.7)
    augmentor.shear(probability=1, max_shear_left=15, max_shear_right=15)
    augmentor.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=3)
    augmentor.skew(probability=0.5g, magnitude=0.3)
    augmentor.resize(1, size[0], size[1])

    augmentor.sample(augmentCount)

    print("\tDone!")
    print("\tReorganizing directory structure...")

    trainingImagesAugmentedPath = outputPath / "images"
    trainingImagesAugmentedPath.mkdir(parents=True, exist_ok=True)
    trainingSegmentationsAugmentedPath = outputPath / "segmentations"
    trainingSegmentationsAugmentedPath.mkdir(parents=True, exist_ok=True)

    results = list(outputPath.glob("*.*"))
    trainingSegmentationFiles = list(outputPath.glob("_groundtruth*"))
    trainingImageFiles = [x for x in results if x not in trainingSegmentationFiles]

    for trainingSegmentationFile in trainingSegmentationFiles:
        newFilename = re.sub(".*_", "", trainingSegmentationFile.name)
        trainingSegmentationFile.rename(trainingSegmentationsAugmentedPath / newFilename)

    for trainingImageFile in trainingImageFiles:
        newFilename = re.sub(".*_", "", trainingImageFile.name)
        trainingImageFile.rename(trainingImagesAugmentedPath / newFilename)

    print("\tDone!")

    results = trainingImagesAugmentedPath, trainingSegmentationsAugmentedPath
    print("Augmented images saved to '" + str(results[0]))
    print("Augmented segmentations saved to '" + str(results[1]))
    print("-----------------------")
