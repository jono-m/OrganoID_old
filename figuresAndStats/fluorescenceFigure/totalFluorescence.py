from pathlib import Path
import sys
import re
import dill

sys.path.append(str(Path(".").resolve()))

import numpy as np
from backend.ImageManager import LoadImages

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3, 1000, 300, 100, 30, 10, 3, np.inf, 0, None]
fluorescenceImages = LoadImages(Path(r"E:\FluoroFiles\*C3*"))
organoidImages = LoadImages(Path(r"E:\FluoroFiles\labeled\*labeled*"))


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
for i, image in enumerate(fluorescenceImages):
    xy = GetXY(image.path.stem)
    print("XY: %d" % xy)
    dosage = GetDosage(xy)
    fluorescences = [np.sum(frame) for frame in image.frames]
    fluorescences = [f / fluorescences[0] for f in fluorescences]
    if GetPatient(xy):
        fluorescenceByDosageA[dosage].append(fluorescences)
    else:
        fluorescenceByDosageB[dosage].append(fluorescences)

areas_by_dosageA = {dosage: [] for dosage in dosages}
numbers_by_dosageA = {dosage: [] for dosage in dosages}
areas_by_dosageB = {dosage: [] for dosage in dosages}
numbers_by_dosageB = {dosage: [] for dosage in dosages}
for i, image in enumerate(organoidImages):
    xy = GetXY(image.path.stem)
    print("XY: %d" % xy)
    dosage = GetDosage(xy)
    areas = [np.count_nonzero(frame) for frame in image.frames]
    areas = [a / areas[0] for a in areas]
    number = [len(np.unique(frame)) - 1 for frame in image.frames]
    number = [n / number[0] for n in number]

    if GetPatient(xy):
        areas_by_dosageA[dosage].append(areas)
        numbers_by_dosageA[dosage].append(number)
    else:
        areas_by_dosageB[dosage].append(areas)
        numbers_by_dosageB[dosage].append(number)

dill.dump(fluorescenceByDosageA, open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceTotalA.pkl", "wb+"))
dill.dump(fluorescenceByDosageB, open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceTotalB.pkl", "wb+"))
dill.dump(areas_by_dosageA, open(r"figuresAndStats\fluorescenceFigure\data\areaTotalA.pkl", "wb+"))
dill.dump(areas_by_dosageB, open(r"figuresAndStats\fluorescenceFigure\data\areaTotalB.pkl", "wb+"))
dill.dump(numbers_by_dosageA, open(r"figuresAndStats\fluorescenceFigure\data\numberTotalA.pkl", "wb+"))
dill.dump(numbers_by_dosageB, open(r"figuresAndStats\fluorescenceFigure\data\numberTotalB.pkl", "wb+"))