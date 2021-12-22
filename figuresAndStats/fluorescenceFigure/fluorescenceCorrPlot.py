from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(".").resolve()))
import re
import matplotlib.pyplot as plt
import colorsys
from backend.ImageManager import LoadImages, ShowImage
from backend.Tracker import Tracker
from skimage.measure import regionprops
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
dosagesToUse = [0, 3, 30, 100, 300, 1000]
tracksByDosage: Dict[int, List[Tuple[int, Tracker.OrganoidTrack]]] = dill.load(file)

times = [18]
dataByDosage = {dosage: [] for dosage in dosagesToUse}
for dose in dosagesToUse:
    for time in times:
        tracksAtTime: List[Tracker.OrganoidTrack] = sum([tracks for xy, tracks in tracksByDosage[dose]], [])
        tracksAtTime = [track for track in tracksAtTime if
                        track.DidTrackExist(time) and track.Data(time).LazyWasDetected()]
        for track in tracksAtTime:
            data = track.Data(time)
            rp, i = data.GetRP()
            fluorescencePerArea = data.extraData['fluorescence'] / rp.area
            circularity = 4 * np.pi * rp.area / (rp.perimeter ** 2)
            dataByDosage[dose].append((data.extraData['fluorescence'], fluorescencePerArea))

dataByDosage = {x: np.asarray(dataByDosage[x]) for x in dataByDosage}
minimum = min([np.min(dataByDosage[x]) for x in dataByDosage])
maximum = max([np.min(dataByDosage[x]) for x in dataByDosage])

for i, dose in enumerate(dosagesToUse):
    plt.subplot(len(dosagesToUse), 1, i+1)
    seaborn.kdeplot(dataByDosage[dose][:, 1])

plt.show()
