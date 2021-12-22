from pathlib import Path
from typing import List
import sys

sys.path.append(str(Path(".").resolve()))
import re
import matplotlib.pyplot as plt
import colorsys
from backend.ImageManager import LoadImages
from backend.Tracker import Tracker
from skimage.measure import regionprops
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


def ParseMap(trackingResultsFile: Path):
    file = open(trackingResultsFile, "r")
    lines = [line.split(", ") for line in file.read().split("\n")[1:-1]]
    file.close()
    labelMaps = []
    for line in lines:
        frameNumber, originalLabel, organoidID = [int(token) for token in line]
        if frameNumber >= len(labelMaps):
            labelMaps.append({})
        labelMaps[frameNumber][originalLabel] = organoidID
    return labelMaps


def BuildTracks(labeledImages, labelMaps):
    builtTracks: List[Tracker.OrganoidTrack] = []

    for n, (labeledImage, labelMap) in enumerate(
            zip(labeledImages, labelMaps)):
        detectedTracks = []
        regionProps = regionprops(labeledImage)
        for rp in regionProps:
            organoidID = labelMap[rp.label]
            matchedTrack = [t for t in builtTracks if t.id == organoidID]
            if not matchedTrack:
                matchedTrack = Tracker.OrganoidTrack(n, organoidID)
                builtTracks.append(matchedTrack)
            else:
                matchedTrack = matchedTrack[0]
                if matchedTrack in detectedTracks:
                    # This one has already been detected!
                    continue
            matchedTrack.Detect(rp)
            detectedTracks.append(matchedTrack)

        [t.NoDetection() for t in builtTracks if t not in detectedTracks]

    return builtTracks


def GetXY(name: str):
    matches = re.findall(r".*XY(\d+).*", name)
    if matches:
        return int(matches[0]) - 1
    return None


def GetDosage(xy: int):
    return dosages[int(xy / 6)]


fluorescenceImages = LoadImages(Path(r"E:\FluoroFiles\*C3*"), size=(512, 512))
organoidImages = LoadImages(Path(r"E:\FluoroFiles\labeled\*labeled*"))
trackingFiles = Path(r"E:\FluoroFiles\tracked").glob("*Results*")
labelMapsByXY = {GetXY(path.stem): ParseMap(path) for path in trackingFiles if GetXY(path.stem) < 54}

maxXY = 54

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3]

tracksByDosage = {dosage: [] for dosage in dosages}
for i, image in enumerate(organoidImages):
    xy = GetXY(image.path.stem)
    if xy >= maxXY:
        break

    dosage = GetDosage(xy)

    newTracks = BuildTracks(image.frames, labelMapsByXY[xy])
    tracksByDosage[dosage].append((xy, newTracks))
    print("XY: %d (%d tracks)" % (xy, len(newTracks)))

for i, image in enumerate(fluorescenceImages):
    xy = GetXY(image.path.stem)
    if xy >= maxXY:
        break

    dosage = GetDosage(xy)

    print("XY: %d" % xy)
    tracks = [tracks for XY, tracks in tracksByDosage[dosage] if XY == xy][0]

    for frameNumber in range(19):
        for trackNumber, track in enumerate(tracks):
            if track.DataAtFrame(frameNumber) and track.LastDetectionFrame() >= frameNumber:
                data = track.DataAtFrame(frameNumber)
                rp = data.regionProperties
                fluorescence = np.sum(image.frames[frameNumber][rp.coords])
                data.fluorescence = fluorescence

outFile = open(Path(r"figuresAndStats\fluorescenceFigure\data\tracks.pkl"), "wb+")
dill.dump(tracksByDosage, outFile)
