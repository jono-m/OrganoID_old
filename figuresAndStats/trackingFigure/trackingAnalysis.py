import sys
from typing import List
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

import matplotlib.pyplot as plt
from skimage.measure import regionprops
from backend.ImageManager import LoadImages, LabelTracks, SaveGIF, SaveImage
from backend.Tracker import Tracker
from scipy.optimize import linear_sum_assignment
import numpy as np


def ParseMap(filename: str):
    labelMap = []
    with open(filename, "r+") as csvFile:
        lines = csvFile.read().split("\n")[1:-1]
        for line in lines:
            line = line.split(", ")
            frameNumber = int(line[0])
            originalLabel = int(line[1])
            organoidID = int(line[2])
            if frameNumber >= len(labelMap):
                labelMap.append({})
            labelMap[frameNumber][originalLabel] = organoidID
    return labelMap


def BuildTracks(allRegionProps, trackMap):
    tracksByID = {}

    nextID = 0
    for frameNumber, (regionProps, labelToTrackMap) in enumerate(
            zip(allRegionProps, trackMap)):
        detectedTracks = []
        for rp in regionProps:
            trackID = labelToTrackMap[rp.label]
            trackID = int(trackID)
            if trackID in tracksByID:
                if tracksByID[trackID] in detectedTracks:
                    # This one has already been detected!
                    continue
            else:
                tracksByID[trackID] = Tracker.OrganoidTrack(frameNumber, nextID)
                nextID += 1
            tracksByID[trackID].Detect(rp.coords)
            detectedTracks.append(tracksByID[trackID])

        [tracksByID[trackID].NoDetection() for trackID in tracksByID if tracksByID[trackID] not in detectedTracks]

    for trackID in tracksByID:
        tracksByID[trackID].id = trackID

    tracksByID = [tracksByID[trackID] for trackID in tracksByID]
    return tracksByID


def MatchTracks(a: List[Tracker.OrganoidTrack], b: List[Tracker.OrganoidTrack]):
    # Matches tracks by the position of their first detected centroid (so we try to match the initial detections
    # and can then compare how each tracks perform over time from there).
    maxSize = max(len(a), len(b))
    costMatrix = np.zeros([maxSize, maxSize])
    for x, organoidA in enumerate(a):
        centroidA = np.array(organoidA.Data(organoidA.firstFrame).GetRP().centroid)
        for y, organoidB in enumerate(b):
            centroidB = np.array(organoidB.Data(organoidB.firstFrame).GetRP().centroid)
            costMatrix[x, y] = np.sqrt(np.sum(np.square(centroidA - centroidB)))

    aIndices, bIndices = linear_sum_assignment(costMatrix)

    matchedA = []
    matchedB = []
    extraA = []
    extraB = []
    for aIndex, bIndex in zip(aIndices, bIndices):
        if aIndex < len(a):
            if bIndex < len(b):
                matchedA.append(a[aIndex])
                matchedB.append(b[bIndex])
            else:
                extraA.append(a[aIndex])
        elif bIndex < len(b):
            extraB.append(b[bIndex])
    return matchedA + extraA, matchedB + extraB


automatedLabelMaps = ParseMap(r"dataset\demo\tracked\trackResults.csv")
groundTruthLabelMaps = ParseMap(r"dataset\demo\tracked\groundTruth\mapping.csv")
labeledImages = \
    list(LoadImages(r"dataset\demo\labeled\20210127_organoidplate003_XY36_Z3_C2_detected_labeled.tiff"))[
        0].frames

originalImage = list(LoadImages(
    r"dataset\demo\20210127_organoidplate003_XY36_Z3_C2.tif", (512, 512), "L"))[0].frames

regionPropsAll = [regionprops(image) for image in labeledImages]

automatedTracks, groundTruthTracks = MatchTracks(BuildTracks(regionPropsAll, automatedLabelMaps),
                                                 BuildTracks(regionPropsAll, groundTruthLabelMaps))

print("Number of Tracks: %d (Auto) and %d (Manual)" % (len(automatedTracks), len(groundTruthTracks)))

maxNumTracks = max(len(automatedTracks), len(groundTruthTracks))
for i in range(maxNumTracks):
    if i < len(automatedTracks):
        automatedTracks[i].id = i
    if i < len(groundTruthTracks):
        groundTruthTracks[i].id = i

color = (255, 255, 255)
idsToHighlight = {0: (0, 154, 222),
                  1: (255, 198, 30),
                  9: (175, 88, 186),
                  33: (0, 205, 108)}
renumberedAutomatedImages = LabelTracks(automatedTracks, (255, 255, 255), 255, 50, color, idsToHighlight,
                                        originalImage)
renumberedGTImages = LabelTracks(groundTruthTracks, (255, 255, 255), 255, 50, color, idsToHighlight,
                                 originalImage)

merged = [np.concatenate([a, b], axis=1) for a, b in zip(renumberedGTImages, renumberedAutomatedImages)]

SaveGIF(merged, Path(r"figuresAndStats\trackingFigure\images\results.gif"))

i = 0
for outputImage in renumberedAutomatedImages:
    fileName = "renumbered_" + str(i) + ".png"
    i += 1
    savePath = Path(r"figuresAndStats\trackingFigure\images") / fileName
    SaveImage(outputImage, savePath)


def CompareTrackData(dataA: Tracker.OrganoidFrameData, dataB: Tracker.OrganoidFrameData):
    if dataA and dataB:
        # Both tracks exist.
        if dataA.WasDetected() and dataB.WasDetected():
            # Both tracks are detecting something in this frame. See if they are tracking the same organoid.
            if dataA.GetRP().label == dataB.GetRP().label:
                rating = 1
            else:
                rating = -1
        elif dataA.WasDetected() == dataB.WasDetected():
            # Both tracks agree that they are missing. But they might be anticipating the same or different organoids.
            rating = 0
        else:
            # One is active and the other is not! Penalize.
            rating = -1
    elif dataA == dataB:
        # Neither track has been detected yet. Not sure how they will do once first detected.
        rating = 0
    else:
        # One has started before the other! Penalize.
        rating = -1

    return rating


numFrames = len(regionPropsAll)
numTracks = max(len(automatedTracks), len(groundTruthTracks))

correctPerFrame = []
incorrectPerFrame = []
percentCorrect = []
areasGT = np.full([numFrames, numTracks], np.nan)
areasAutomated = np.full([numFrames, numTracks], np.nan)

for frameNumber in range(numFrames):
    correctTracks = 0
    incorrectTracks = 0
    for trackNumber, (automatedTrack, groundTruthTrack) in enumerate(zip(automatedTracks, groundTruthTracks)):
        comparison = CompareTrackData(automatedTrack.Data(frameNumber),
                                      groundTruthTrack.Data(frameNumber))
        if comparison == 1:
            correctTracks += 1
        elif comparison == -1:
            incorrectTracks += 1

        if automatedTrack.WasDetected(frameNumber) and automatedTrack.GetLastDetectedFrame() > frameNumber:
            areasAutomated[frameNumber, trackNumber] = automatedTrack.Data(
                frameNumber).GetRP().area * 6.8644 / 1000

        if groundTruthTrack.WasDetected(frameNumber) and groundTruthTrack.GetLastDetectedFrame() > frameNumber:
            areasGT[frameNumber, trackNumber] = groundTruthTrack.Data(
                frameNumber).GetRP().area * 6.8644 / 1000

    correctPerFrame.append(correctTracks)
    incorrectPerFrame.append(incorrectTracks)
    percentCorrect.append(correctTracks / (correctTracks + incorrectTracks))

performance = np.stack([np.asarray(x) for x in [correctPerFrame, incorrectPerFrame, percentCorrect]])
np.savetxt(r"figuresAndStats\trackingFigure\data\performance.dat", performance)
np.savetxt(r"figuresAndStats\trackingFigure\data\areasGT.dat", areasGT)
np.savetxt(r"figuresAndStats\trackingFigure\data\areasAutomated.dat", areasAutomated)
