from pathlib import Path
import sys
import re
import matplotlib.pyplot as plt
import colorsys

sys.path.append(str(Path(".").resolve()))

import numpy as np
from backend.ImageManager import LoadImages

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
meanColor = [x / 255 for x in (0, 154, 222)]
lodColor = [x / 255 for x in (255, 31, 91)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3]
fluorescenceImages = LoadImages(Path(r"E:\FluoroFiles\*C3*"))
organoidImages = LoadImages(Path(r"E:\FluoroFiles\labeled\*labeled*"))


def GetXY(name: str):
    matches = re.findall(r".*XY(\d+).*", name)
    if matches:
        return int(matches[0]) - 1
    return None


def GetDosage(xy: int):
    return dosages[int(xy / 6)]


fluorescenceByDosage = {dosage: [] for dosage in dosages}
for i, image in enumerate(fluorescenceImages):
    xy = GetXY(image.path.stem)
    if xy >= 54:
        break
    print("XY: %d" % xy)
    dosage = GetDosage(xy)
    fluorescences = [np.sum(frame) for frame in image.frames]
    fluorescences = [f / fluorescences[0] for f in fluorescences]
    fluorescenceByDosage[dosage].append(fluorescences)

areas_by_dosage = {dosage: [] for dosage in dosages}
numbers_by_dosage = {dosage: [] for dosage in dosages}
for i, image in enumerate(organoidImages):
    xy = GetXY(image.path.stem)
    if xy >= 54:
        break
    print("XY: %d" % xy)
    dosage = GetDosage(xy)
    areas = [np.count_nonzero(frame) for frame in image.frames]
    areas = [a / areas[0] for a in areas]
    areas_by_dosage[dosage].append(areas)
    number = [len(np.unique(frame)) - 1 for frame in image.frames]
    number = [n / number[0] for n in number]
    numbers_by_dosage[dosage].append(number)

dosagesToUse = [0, 3, 10, 30, 100, 300, 1000]
hue = 343 / 360
colors = [colorsys.hsv_to_rgb(hue, sat, 1) for sat in np.linspace(0.1, 1, 4)] + \
         [colorsys.hsv_to_rgb(hue, 1, val) for val in np.linspace(0.8, 0, 3)]

for color, dosage in zip(colors, dosagesToUse):
    label = "%d nM" % dosage
    fluorescences = np.asarray(fluorescenceByDosage[dosage])
    areas = np.asarray(areas_by_dosage[dosage])
    numbers = np.asarray(numbers_by_dosage[dosage])
    plt.subplot(2, 2, 1)
    plt.errorbar(np.arange(0, 73, 4), np.mean(areas, axis=0),
                 yerr=np.std(areas, axis=0) / np.sqrt(areas.shape[0]),
                 label=label, color=color)

    areas = np.asarray(areas_by_dosage[dosage])
    plt.subplot(2, 2, 2)
    plt.errorbar(np.arange(0, 73, 4), np.mean(numbers, axis=0),
                 yerr=np.std(numbers, axis=0) / np.sqrt(areas.shape[0]),
                 label=label, color=color)

    plt.subplot(2, 2, 3)
    plt.errorbar(np.arange(0, 73, 4), np.mean(fluorescences, axis=0),
                 yerr=np.std(fluorescences, axis=0) / np.sqrt(fluorescences.shape[0]),
                 label=label, color=color)

    plt.subplot(2, 2, 4)
    fluorescencePerArea = np.divide(fluorescences, areas)
    plt.errorbar(np.arange(0, 73, 4), np.mean(fluorescencePerArea, axis=0),
                 yerr=np.std(fluorescencePerArea, axis=0) / np.sqrt(fluorescencePerArea.shape[0]),
                 label=label, color=color)

plt.subplot(2, 2, 1)
plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel("Organoid area (fold change from t=0)")

plt.subplot(2, 2, 2)
plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel("Number of organoids (fold change from t=0)")

plt.subplot(2, 2, 3)
plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel("Fluorescent Death Signal (fold change from t=0)")

plt.subplot(2, 2, 4)
plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel("Fluorescent Death Signal per Area (fold change from t=0)")
plt.show()
