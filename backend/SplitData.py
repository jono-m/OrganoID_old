from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
import shutil


def SplitData(imagePaths: List[Path], segmentationPaths: List[Path], testSize: float,
              outputPath: Path = None):
    imagePaths.sort(key=lambda x: x.stem)
    segmentationPaths.sort(key=lambda x: x.stem)

    trainingImagePaths, testingImagePaths, trainingSegmentationPaths, testingSegmentationPaths = train_test_split(
        imagePaths,
        segmentationPaths,
        test_size=testSize)

    if outputPath is not None:
        _CopyToPath(trainingImagePaths, outputPath / "training" / "images")
        _CopyToPath(testingImagePaths, outputPath / "validation" / "segmentations")
        _CopyToPath(trainingSegmentationPaths, outputPath / "training" / "images")
        _CopyToPath(testingSegmentationPaths, outputPath / "validation" / "segmentations")


def _CopyToPath(paths: List[Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        newPath = output / path.name
        shutil.copy(path, newPath)
