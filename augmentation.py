import Augmentor
from pathlib import Path
import re
import typing


def AugmentImages(trainingImagesPath: Path, trainingSegmentationsPath: Path,
                  outputPath: Path, samplesToTake: int,
                  size: typing.Tuple[int, int]) -> typing.Tuple[Path, Path]:
    print("-----------------------")
    print("Augmenting training data...")
    print("\tImages directory: " + str(trainingImagesPath))
    print("\tSegmentations directory: " + str(trainingSegmentationsPath))

    augmentor = Augmentor.Pipeline(source_directory=trainingImagesPath,
                                   output_directory=outputPath)
    augmentor.set_save_format("auto")
    augmentor.ground_truth(trainingSegmentationsPath)

    augmentor.resize(1, size[0], size[1])
    augmentor.greyscale(probability=1)
    augmentor.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    augmentor.flip_left_right(probability=0.5)
    augmentor.flip_top_bottom(probability=0.5)
    augmentor.zoom_random(probability=0.5, percentage_area=0.3, randomise_percentage_area=True)
    augmentor.shear(probability=0.5, max_shear_left=14, max_shear_right=15)
    augmentor.random_distortion(probability=0.2, grid_width=5, grid_height=7, magnitude=3)
    augmentor.skew(probability=0.8, magnitude=0.4)
    augmentor.resize(1, size[0], size[1])
    augmentor.sample(samplesToTake)

    print("\tDone!")
    print("\tReorganizing directory structure...")

    trainingImagesAugmentedPath = outputPath / "images"
    trainingImagesAugmentedPath.mkdir()
    trainingSegmentationsAugmentedPath = outputPath / "segmentations"
    trainingSegmentationsAugmentedPath.mkdir()

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

    return results
