from PIL import Image
from pathlib import Path
import numpy as np


def ComputeClassImbalance(segmentationsDirectory: Path):
    files = segmentationsDirectory.iterdir()
    totalZeros = 0
    totalOnes = 0
    for fileIndex, fileName in enumerate(files):
        segmentation = Image.open(fileName)
        image = np.array(segmentation.convert(mode="1"))
        size = np.size(image)
        ones = np.count_nonzero(image)
        zeros = size - ones
        totalZeros += zeros
        totalOnes += ones
        print(fileIndex)
    print("Zeros: " + str(totalZeros), ". Ones: " + str(totalZeros) + ". Fraction: " + str(totalOnes / totalZeros))g


path = Path("/home/jono/ML/TrainData/OrganoID_augment_2021_06_15_15_24_28/segmentations/")
ComputeClassImbalance(path)
