# SplitData.py -- split image/segmentation pairs into training and validation sets.

from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
import shutil


def SplitData(imagePaths: List[Path], segmentationPaths: List[Path], testSize: float,
              outputPath: Path = None):
    # Sort paths alphabetically
    imagePaths.sort(key=lambda x: x.stem)
    segmentationPaths.sort(key=lambda x: x.stem)

    # Only use paths that have matched names in image and segmentations directory
    imagePaths = [path for path in imagePaths if path.stem in [segPath.stem for segPath in segmentationPaths]]
    segmentationPaths = [path for path in segmentationPaths if
                         path.stem in [imagePath.stem for imagePath in imagePaths]]

    # Carry out the split!
    trainingImagePaths, testingImagePaths, trainingSegmentationPaths, testingSegmentationPaths = train_test_split(
        imagePaths,
        segmentationPaths,
        test_size=testSize)

    # Save the images
    if outputPath is not None:
        _CopyToPath(trainingImagePaths, outputPath / "training" / "images")
        _CopyToPath(testingImagePaths, outputPath / "validation" / "images")
        _CopyToPath(trainingSegmentationPaths, outputPath / "training" / "segmentations")
        _CopyToPath(testingSegmentationPaths, outputPath / "validation" / "segmentations")


def _CopyToPath(paths: List[Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        newPath = output / path.name
        shutil.copy(path, newPath)
