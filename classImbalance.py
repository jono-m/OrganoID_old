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
    print("Zeros: " + str(totalZeros), ". Ones: " + str(totalOnes) + ". Fraction: " + str(totalOnes / totalZeros))


path = Path(r"C:\Users\jonoj\Documents\ML\Jono_2021Masks\202105_OriginalY")
ComputeClassImbalance(path)
