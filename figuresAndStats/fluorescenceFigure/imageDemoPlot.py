from pathlib import Path
from typing import List
import sys

sys.path.append(str(Path(".").resolve()))
import re
from backend.ImageManager import LoadImages, ComputeOutline, SaveImage, ShowImage
from PIL import Image, ImageDraw
from skimage.measure import regionprops
import numpy as np


def GetXY(name: str):
    matches = re.findall(r".*XY(\d+).*", name)
    if matches:
        return int(matches[0]) - 1
    return None


def GetDosage(xy: int):
    return dosages[int(xy / 6)]


originalImages = LoadImages(Path(r"E:\FluoroFiles\*C1*"), mode="L", size=(512, 512))
fluorescenceImages = LoadImages(Path(r"E:\FluoroFiles\*C3*"), mode="L", size=(512, 512))
labeledImages = LoadImages(Path(r"E:\FluoroFiles\labeled\*labeled*"))

maxXY = 54

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3]

times = [0, 9, 18]

dosagesToUse = {0: 0,
                30: 0,
                1000: 0}

originalImageFrames = {}
for i, image in enumerate(originalImages):
    xy = GetXY(image.path.stem)
    print("O XY: %d" % xy)
    if xy >= maxXY:
        break

    dosage = GetDosage(xy)

    if dosage in dosagesToUse and dosage not in originalImageFrames:
        if (xy - 1) % 6 >= dosagesToUse[dosage]:
            # Use
            originalImageFrames[dosage] = [image.frames[time] for time in times]

    if len([dosage for dosage in dosagesToUse if dosage not in originalImageFrames]) == 0:
        break

fluorescenceImageFrames = {}
for i, image in enumerate(fluorescenceImages):
    xy = GetXY(image.path.stem)
    print("F XY: %d" % xy)
    if xy >= maxXY:
        break

    dosage = GetDosage(xy)

    if dosage in dosagesToUse and dosage not in fluorescenceImageFrames:
        if (xy - 1) % 6 >= dosagesToUse[dosage]:
            # Use
            fluorescenceImageFrames[dosage] = [image.frames[time] for time in times]

    if len([dosage for dosage in dosagesToUse if dosage not in fluorescenceImageFrames]) == 0:
        break

labeledImageFrames = {}
for i, image in enumerate(labeledImages):
    xy = GetXY(image.path.stem)
    print("L XY: %d" % xy)
    if xy >= maxXY:
        break

    dosage = GetDosage(xy)

    if dosage in dosagesToUse and dosage not in labeledImageFrames:
        if (xy - 1) % 6 >= dosagesToUse[dosage]:
            # Use
            labeledImageFrames[dosage] = [image.frames[time] for time in times]

    if len([dosage for dosage in dosagesToUse if dosage not in labeledImageFrames]) == 0:
        break


# Overlay a set of organoid tracks on a list of base images.
def BuildImages(originalImage, fluorescenceImage, labeledImage):
    fluorescenceImageNormalized = (fluorescenceImage - np.min(fluorescenceImage)) / (
            np.max(fluorescenceImage) - np.min(fluorescenceImage)) * 2
    fluorescenceImageNormalized = np.where(fluorescenceImageNormalized > 1, 1, fluorescenceImageNormalized)

    fluorescenceImageTransparent = np.multiply(fluorescenceImageNormalized[:, :, None],
                                               np.array([255, 31, 91, 255])[None, None, :]).astype(np.uint8)
    fluorescenceImageTransparent = Image.fromarray(fluorescenceImageTransparent)
    originalFluorescenceOverlay = Image.alpha_composite(Image.fromarray(originalImage).convert(mode="RGBA"),
                                                        fluorescenceImageTransparent).convert(mode="RGB")

    fluorescenceImageColored = np.multiply(fluorescenceImageNormalized[:, :, None],
                                           np.array([255, 31, 91])[None, None, :]).astype(
        np.uint8)
    fluorescenceImageColored = Image.fromarray(fluorescenceImageColored).convert(mode="RGBA")

    labeledFluorescenceImage = fluorescenceImageColored.copy()
    drawer = ImageDraw.Draw(labeledFluorescenceImage)
    rps = regionprops(labeledImage)
    for rp in rps:
        borderCoords = ComputeOutline(rp.image)
        globalCoords = borderCoords + rp.bbox[:2]
        xs = list(globalCoords[:, 1])
        ys = list(globalCoords[:, 0])
        outlineCoords = list(zip(xs, ys))
        drawer.point(outlineCoords, (255, 255, 255, 255))

    return originalFluorescenceOverlay, labeledFluorescenceImage


for dosage in dosagesToUse:
    for i, time in enumerate(times):
        ofo, lf = BuildImages(originalImageFrames[dosage][i],
                              fluorescenceImageFrames[dosage][i],
                              labeledImageFrames[dosage][i])
        SaveImage(np.asarray(ofo),
                  Path(r"figuresAndStats\fluorescenceFigure\images\bfFluorescence_%d_%d.png" % (dosage, time)))

        SaveImage(np.asarray(lf),
                  Path(r"figuresAndStats\fluorescenceFigure\images\labeledFluorescence_%d_%d.png" % (dosage, time)))
