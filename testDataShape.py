from PIL import Image
from pathlib import Path
import numpy as np
import sys

trainingImagesPath = "C:/Users/jonoj/Documents/ML/2019_originalX/OrganoID_aug_2021_03_02_09_56_39"
trainingSegmentationsPath = "C:/Users/jonoj/Documents/ML/2019_originalY/OrganoID_aug_2021_03_02_09_56_39"

images = [Image.open(imagePath) for imagePath in sorted(Path(trainingImagesPath).iterdir()) if imagePath.is_file()]

segmentations = [Image.open(segmentationPath) for segmentationPath in sorted(Path(trainingSegmentationsPath).iterdir())
                 if
                 segmentationPath.is_file()]

imageSize = images[0].size

for imageIndex, image in enumerate(images):
    if image.mode == 'I':
        image = image.point(lambda x: x*(1/255))
    images[imageIndex] = np.array(image.resize(imageSize).convert(mode="RGB"))

images = np.stack(images, axis=-1)

for segmentationindex, segmentation in enumerate(segmentations):
    segmentations[segmentationindex] = np.array(segmentation.resize(imageSize).convert(mode="1"))

segmentations = np.stack(segmentations, axis=-1)

sys.exit()
