from pathlib import Path
import sys
import re
import dill

sys.path.append(str(Path(".").resolve()))

import numpy as np
from PIL import Image
from backend.ImageManager import LoadImages, ShowImage

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3, 1000, 300, 100, 30, 10, 3, np.inf, 0, None]
fluorescenceImages = LoadImages(Path(r"E:\FluoroFiles\*C3*"))
labeledImages = LoadImages(Path(r"E:\FluoroFiles\labeled\*labeled*"))


def GetXY(name: str):
    matches = re.findall(r".*XY(\d+).*", name)
    if matches:
        return int(matches[0]) - 1
    return None


def GetDosage(xy: int):
    return dosages[int(xy / 6)]


def GetPatient(xy: int):
    return xy >= 54


fluorescenceByDosageA = {dosage: [] for dosage in dosages}
fluorescenceByDosageB = {dosage: [] for dosage in dosages}
fluorescenceInRegionA = {dosage: [] for dosage in dosages}
fluorescenceInRegionB = {dosage: [] for dosage in dosages}
areas_by_dosageA = {dosage: [] for dosage in dosages}
numbers_by_dosageA = {dosage: [] for dosage in dosages}
areas_by_dosageB = {dosage: [] for dosage in dosages}
numbers_by_dosageB = {dosage: [] for dosage in dosages}
for i, (fluorescenceImage, labeledImage) in enumerate(zip(fluorescenceImages, labeledImages)):
    xy = GetXY(labeledImage.path.stem)
    xy2 = GetXY(fluorescenceImage.path.stem)
    if xy != xy2:
        print("BAD MATCH")
        sys.exit()
    print("XY: %d" % xy)
    dosage = GetDosage(xy)
    areas = [np.count_nonzero(frame) for frame in labeledImage.frames]
    number = [len(np.unique(frame)) - 1 for frame in labeledImage.frames]
    fluorescences = [np.sum(frame / 1000) for frame in fluorescenceImage.frames]
    regionFluorescences = []
    for fluorescenceFrame, labeledFrame in zip(fluorescenceImage.frames, labeledImage.frames):
        scaledLabeledFrame = np.asarray(Image.fromarray(labeledFrame == 0).resize(fluorescenceFrame.shape, Image.NEAREST))
        ff = fluorescenceFrame / 1000
        ff[scaledLabeledFrame] = 0
        regionFluorescences.append(np.sum(ff))
    if GetPatient(xy):
        if xy % 2 == 1:
            fluorescenceByDosageA[dosage][-1] = list(
                np.array(fluorescenceByDosageA[dosage][-1]) + np.array(fluorescences))
            fluorescenceInRegionA[dosage][-1] = list(
                np.array(fluorescenceInRegionA[dosage][-1]) + np.array(regionFluorescences))
            areas_by_dosageA[dosage][-1] = list(np.array(areas_by_dosageA[dosage][-1]) + np.array(areas))
            numbers_by_dosageA[dosage][-1] = list(np.array(numbers_by_dosageA[dosage][-1]) + np.array(number))
        else:
            fluorescenceByDosageA[dosage].append(fluorescences)
            fluorescenceInRegionA[dosage].append(regionFluorescences)
            areas_by_dosageA[dosage].append(areas)
            numbers_by_dosageA[dosage].append(number)
    else:
        if xy % 2 == 1:
            fluorescenceByDosageB[dosage][-1] = list(
                np.array(fluorescenceByDosageB[dosage][-1]) + np.array(fluorescences))
            fluorescenceInRegionB[dosage][-1] = list(
                np.array(fluorescenceInRegionB[dosage][-1]) + np.array(regionFluorescences))
            areas_by_dosageB[dosage][-1] = list(np.array(areas_by_dosageB[dosage][-1]) + np.array(areas))
            numbers_by_dosageB[dosage][-1] = list(np.array(numbers_by_dosageB[dosage][-1]) + np.array(number))
        else:
            fluorescenceByDosageB[dosage].append(fluorescences)
            fluorescenceInRegionB[dosage].append(regionFluorescences)
            areas_by_dosageB[dosage].append(areas)
            numbers_by_dosageB[dosage].append(number)

dill.dump(fluorescenceByDosageA, open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceTotalA.pkl", "wb+"))
dill.dump(fluorescenceByDosageB, open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceTotalB.pkl", "wb+"))
dill.dump(fluorescenceInRegionA, open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceInRegionA.pkl", "wb+"))
dill.dump(fluorescenceInRegionB, open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceInRegionB.pkl", "wb+"))
dill.dump(areas_by_dosageA, open(r"figuresAndStats\fluorescenceFigure\data\areaTotalA.pkl", "wb+"))
dill.dump(areas_by_dosageB, open(r"figuresAndStats\fluorescenceFigure\data\areaTotalB.pkl", "wb+"))
dill.dump(numbers_by_dosageA, open(r"figuresAndStats\fluorescenceFigure\data\numberTotalA.pkl", "wb+"))
dill.dump(numbers_by_dosageB, open(r"figuresAndStats\fluorescenceFigure\data\numberTotalB.pkl", "wb+"))
