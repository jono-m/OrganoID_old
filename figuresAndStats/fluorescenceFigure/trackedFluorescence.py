from pathlib import Path
from typing import List
import sys

sys.path.append(str(Path(".").resolve()))
import re
from backend.ImageManager import LoadImages
from backend.Tracker import Tracker
from skimage.measure import regionprops
import dill
import numpy as np


def ParseMap(trackingResultsFile: Path):
    file = open(trackingResultsFile, "r")
    lines = [line.split(", ") for line in file.read().split("\n")[1:-1]]
    file.close()
    labelMaps = [{} for _ in range(19)]
    for line in lines:
        frameNumber, originalLabel, organoidID = [int(token) for token in line]
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
            matchedTrack.Detect(rp.coords)
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


def GetPatient(xy: int):
    return xy >= 54


fluorescenceImages = LoadImages(Path(r"E:\FluoroFiles\*C3*"))
organoidImages = LoadImages(Path(r"E:\FluoroFiles\labeled\*labeled*"))
trackingFiles = Path(r"E:\FluoroFiles\tracked").glob("*Results*")
labelMapsByXY = {GetXY(path.stem): ParseMap(path) for path in trackingFiles}

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3, 1000, 300, 100, 30, 10, 3, np.inf, 0, None]

tracksByDosageA = {dosage: [] for dosage in dosages}
tracksByDosageB = {dosage: [] for dosage in dosages}
for i, image in enumerate(organoidImages):
    xy = GetXY(image.path.stem)
    dosage = GetDosage(xy)

    newTracks = BuildTracks(image.frames, labelMapsByXY[xy])
    if GetPatient(xy):
        tracksByDosageA[dosage].append((xy, newTracks))
    else:
        tracksByDosageB[dosage].append((xy, newTracks))
    print("XY: %d (%d tracks)" % (xy, len(newTracks)))

for i, image in enumerate(fluorescenceImages):
    xy = GetXY(image.path.stem)
    dosage = GetDosage(xy)

    print("XY: %d" % xy)
    if GetPatient(xy):
        tracksByDosage = tracksByDosageA
    else:
        tracksByDosage = tracksByDosageB
    tracks = [tracks for XY, tracks in tracksByDosage[dosage] if XY == xy][0]

    for frameNumber in range(19):
        for trackNumber, track in enumerate(tracks):
            if track.WasDetected(frameNumber) and track.GetLastDetectedFrame() >= frameNumber:
                data = track.Data(frameNumber)
                rp = data.GetRP()
                fluorescence = np.sum(image.frames[frameNumber][rp.coords]/1000)
                data.extraData['fluorescence'] = fluorescence

dill.dump(tracksByDosageA, open(Path(r"figuresAndStats\fluorescenceFigure\data\tracksA.pkl"), "wb+"))
dill.dump(tracksByDosageB, open(Path(r"figuresAndStats\fluorescenceFigure\data\tracksB.pkl"), "wb+"))
