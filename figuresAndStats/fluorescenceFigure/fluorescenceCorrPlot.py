from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import colorsys
from backend.Tracker import Tracker
from scipy.stats import f_oneway, kruskal, alexandergovern
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
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
dosagesToUse = [0, 10, 100]
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

d = allData[allData["Time"] == 18]
jp = seaborn.JointGrid(data=d, y="Fluorescence intensity per area", x="Circularity", hue="Dose",
                       palette={dose: color for color, dose in zip(colors, dosagesToUse)})
jp.plot_marginals(seaborn.kdeplot, common_norm=False, fill=True, alpha=0.7)
jp.plot_joint(seaborn.scatterplot, s=10)
jp.ax_joint.axvline(1.0, color="black")
for color, dose in zip(colors, dosagesToUse):
    jp.ax_marg_x.axvline(d[d["Dose"] == dose]["Circularity"].mean(), color=color)
    jp.ax_marg_y.axhline(d[d["Dose"] == dose]["Fluorescence intensity per area"].mean(), color=color)

fig, axes = plt.subplots(1, 2)
seaborn.kdeplot(data=d, x="Fluorescence", hue="Dose", palette={dose: color for color, dose in zip(colors, dosagesToUse)},
                common_norm=False, fill=True, alpha=0.7, ax=axes[0])
for color, dose in zip(colors, dosagesToUse):
    axes[0].axvline(d[d["Dose"] == dose]["Fluorescence"].mean(), color=color)

seaborn.kdeplot(data=d, x="Area", hue="Dose", palette={dose: color for color, dose in zip(colors, dosagesToUse)},
                common_norm=False, fill=True, alpha=0.7, ax=axes[1])
for color, dose in zip(colors, dosagesToUse):
    axes[1].axvline(d[d["Dose"] == dose]["Area"].mean(), color=color)
circularitySamples = [d[d["Dose"] == dose]["Circularity"] for dose in dosagesToUse]
fpaSamples = [d[d["Dose"] == dose]["Fluorescence intensity per area"] for dose in dosagesToUse]
fSamples = [d[d["Dose"] == dose]["Fluorescence"] for dose in dosagesToUse]
aSamples = [d[d["Dose"] == dose]["Area"] for dose in dosagesToUse]
print("----\nANOVA\n----")
print("Circularity: " + str(f_oneway(*circularitySamples)))
print("FPA: " + str(f_oneway(*fpaSamples)))
print("Fluorescence: " + str(f_oneway(*fSamples)))
print("Area: " + str(f_oneway(*aSamples)))

# plt.legend()
plt.show()
