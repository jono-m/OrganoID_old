from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(".").resolve()))
import re
import matplotlib.pyplot as plt
import colorsys
from backend.ImageManager import LoadImages, ShowImage
from backend.Tracker import Tracker
import scipy.stats as stats
from skimage.measure import regionprops
import pandas
import seaborn
import dill
import numpy as np

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

file = open(Path(r"figuresAndStats\fluorescenceFigure\data\tracksB.pkl"), "rb")
dosagesToUse = [0]
tracksByDosage: Dict[int, List[Tuple[int, Tracker.OrganoidTrack]]] = dill.load(file)

allData = []
for dose in dosagesToUse:
    tracks: List[Tracker.OrganoidTrack] = sum([tracks for xy, tracks in tracksByDosage[dose]], [])
    data = []

    for trackNo, track in enumerate(tracks):
        for time in range(19):
            if not track.WasDetected(time):
                continue

            fluorescence = track.Data(time).extraData["fluorescence"]
            area = track.Data(time).GetRP().area
            circularity = (2 * np.pi * (track.Data(time).GetRP().equivalent_diameter / 2 - 0.5)) / track.Data(
                time).GetRP().perimeter
            data.append((dose, time, circularity, fluorescence, area))

    data = pandas.DataFrame(data, columns=["Dose", "Time", "Circularity", "Fluorescence", "Area"])
    allData.append(data)
    print(dose)

allData = pandas.concat(allData, ignore_index=True)

hue = 343 / 360
colors = [colorsys.hsv_to_rgb(hue, sat, 1) for sat in np.linspace(0.2, 1, 2)] + \
         [colorsys.hsv_to_rgb(hue, 1, val) for val in np.linspace(0, 0, 1)]
allData["Fluorescence intensity per area"] = allData["Fluorescence"] / allData["Area"]

circBounds = [f(allData["Circularity"]) for f in (np.min, np.max)]
fluorBounds = [f(allData["Fluorescence intensity per area"]) for f in (np.min, np.max)]

d = allData[allData["Time"] == 18]
jp = seaborn.JointGrid(data=d, y="Fluorescence intensity per area", x="Circularity", hue="Dose",
                       palette={dose: color for color, dose in zip(colors, dosagesToUse)})
jp.plot_marginals(seaborn.kdeplot, common_norm=False, fill=True, alpha=0.7)
jp.plot_joint(seaborn.kdeplot, common_norm=False)
jp.plot_joint(seaborn.scatterplot, s=4)


# plt.legend()
plt.show()
