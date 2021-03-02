import Augmentor
from pathlib import Path
import re


def AugmentImages(jobID: str, trainingImagesPath: str, trainingSegmentationsPath: str, samplesToTake: int):
    print("-----------------------")
    print("Augmenting training data...")
    print("\tImages directory: " + trainingImagesPath)
    print("\tSegmentations directory: " + trainingSegmentationsPath)

    outputDirectory = 'OrganoID_aug_' + jobID

    def PrepareAugmentor():
        augmentor = Augmentor.Pipeline(source_directory=trainingImagesPath,
                                       output_directory=outputDirectory)
        augmentor.set_save_format("auto")
        augmentor.ground_truth(trainingSegmentationsPath)
        return augmentor

    p = PrepareAugmentor()
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.crop_random(probability=1, percentage_area=0.9)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.flip_left_right(probability=1)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.zoom_random(probability=1, percentage_area=0.8)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.zoom_random(probability=1, percentage_area=0.7)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.flip_top_bottom(probability=1)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.shear(probability=1, max_shear_left=14, max_shear_right=15)
    p.random_distortion(probability=0.2, grid_width=5, grid_height=7, magnitude=3)
    p.skew(probability=0.8, magnitude=0.4)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.shear(probability=0.4, max_shear_left=14, max_shear_right=15)
    p.random_distortion(probability=0.2, grid_width=5, grid_height=7, magnitude=3)
    p.skew(probability=1, magnitude=0.4)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.shear(probability=0.4, max_shear_left=14, max_shear_right=15)
    p.random_distortion(probability=1, grid_width=5, grid_height=7, magnitude=3)
    p.skew(probability=0.4, magnitude=0.4)
    p.sample(samplesToTake)

    p = PrepareAugmentor()
    p.zoom(probability=1, min_factor=1.1, max_factor=1.4)
    p.sample(samplesToTake)

    trainingImagesAugmentedPath = Path(trainingImagesPath).resolve() / outputDirectory
    trainingSegmentationsAugmentedPath = Path(trainingSegmentationsPath).resolve() / outputDirectory
    trainingSegmentationsAugmentedPath.mkdir(exist_ok=True)

    results = list(trainingImagesAugmentedPath.glob("*.*"))
    trainingSegmentationFiles = list(trainingImagesAugmentedPath.glob("_groundtruth*"))
    trainingImageFiles = [x for x in results if x not in trainingSegmentationFiles]

    print("\tDone!")
    print("\tReorganizing directory structure...")

    for trainingSegmentationFile in trainingSegmentationFiles:
        newFilename = re.sub(".*_", "", trainingSegmentationFile.name)
        trainingSegmentationFile.rename(trainingSegmentationsAugmentedPath / newFilename)

    for trainingImageFile in trainingImageFiles:
        newFilename = re.sub(".*_", "", trainingImageFile.name)
        trainingImageFile.rename(trainingImagesAugmentedPath / newFilename)

    print("\tDone!")

    results = str(trainingImagesAugmentedPath.resolve()), str(trainingSegmentationsAugmentedPath.resolve())
    print("Augmented images saved to '" + results[0])
    print("Augmented segmentations saved to '" + results[1])
    print("-----------------------")

    return results
