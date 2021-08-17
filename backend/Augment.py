import Augmentor
from pathlib import Path
import re


def Augment(imagesPath: Path, segmentationsPath: Path, outputPath: Path, count: int):
    outputPath.mkdir(parents=True, exist_ok=True)

    augmentor = Augmentor.Pipeline(source_directory=imagesPath,
                                   output_directory=outputPath)
    augmentor.set_save_format("auto")
    augmentor.ground_truth(segmentationsPath)

    augmentor.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    augmentor.flip_left_right(probability=0.5)
    augmentor.flip_top_bottom(probability=0.5)
    augmentor.zoom_random(probability=0.5, percentage_area=0.7)
    augmentor.shear(probability=1, max_shear_left=20, max_shear_right=20)
    augmentor.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=3)
    augmentor.skew(probability=0.5, magnitude=0.3)
    augmentor.resize(1, 512, 512)

    augmentor.sample(count)

    outputImagesPath = outputPath / "images"
    outputImagesPath.mkdir(parents=True, exist_ok=True)
    outputSegmentationsPath = outputPath / "segmentations"
    outputSegmentationsPath.mkdir(parents=True, exist_ok=True)

    files = [path for path in outputPath.iterdir() if path.is_file()]
    segmentationFiles = [path for path in files if re.match("_groundtruth*", path.stem)]
    trainingImageFiles = [x for x in files if x not in segmentationFiles]

    for segmentationFile in segmentationFiles:
        newFilename = re.sub(".*_", "", segmentationFile.name)
        segmentationFile.rename(outputSegmentationsPath / newFilename)

    for imageFile in trainingImageFiles:
        newFilename = re.sub(".*_", "", imageFile.name)
        imageFile.rename(outputImagesPath / newFilename)
