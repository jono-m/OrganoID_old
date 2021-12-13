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

plt.rcParams['svg.fonttype'] = 'none'


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
                tracksByID[trackID] = Tracker.OrganoidTrack(frameNumber)
            tracksByID[trackID].Detect(rp.centroid, rp.area, rp.coords, rp.image, rp.bbox, rp.label)
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
        centroidA = np.array(organoidA.data[0].centroid)
        for y, organoidB in enumerate(b):
            centroidB = np.array(organoidB.data[0].centroid)
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


renumberedAutomatedImages = LabelTracks(automatedTracks, (255, 255, 255), 255, 100, (0, 255, 0), (255, 0, 0),
                                        originalImage)
renumberedGTImages = LabelTracks(groundTruthTracks, (255, 255, 255), 255, 100, (0, 255, 0), (255, 0, 0),
                                 originalImage)

merged = [np.concatenate([a, b], axis=1) for a, b in zip(renumberedGTImages, renumberedAutomatedImages)]

SaveGIF(merged, Path(r"figuresAndStats\trackingFigure\images\renumberedTracks.gif"))

i = 0
for outputImage in merged:
    fileName = "trackResults_" + str(i) + ".png"
    i += 1
    savePath = Path(r"figuresAndStats\trackingFigure\images") / fileName
    SaveImage(outputImage, savePath)


def CompareTrackData(dataA: Tracker.OrganoidFrameData, dataB: Tracker.OrganoidFrameData):
    if dataA and dataB:
        # Both tracks exist.
        if dataA.wasDetected and dataB.wasDetected:
            # Both tracks are detecting something in this frame. See if they are tracking the same organoid.
            if dataA.label == dataB.label:
                rating = 1
            else:
                rating = -1
        elif dataA.wasDetected == dataB.wasDetected:
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
        comparison = CompareTrackData(automatedTrack.DataAtFrame(frameNumber),
                                      groundTruthTrack.DataAtFrame(frameNumber))
        if comparison == 1:
            correctTracks += 1
        elif comparison == -1:
            incorrectTracks += 1

        if automatedTrack.DataAtFrame(frameNumber):
            areasAutomated[frameNumber, trackNumber] = automatedTrack.DataAtFrame(frameNumber).area * 6.8644 / 1000

        if groundTruthTrack.DataAtFrame(frameNumber):
            areasGT[frameNumber, trackNumber] = groundTruthTrack.DataAtFrame(frameNumber).area * 6.8644 / 1000

    correctPerFrame.append(correctTracks)
    incorrectPerFrame.append(incorrectTracks)
    percentCorrect.append(correctTracks / (correctTracks + incorrectTracks))

frames = [i * 2 for i in range(numFrames)]
plt.subplot(2, 2, 1)
plt.plot(frames, correctPerFrame)
plt.plot(frames, incorrectPerFrame)
plt.plot(frames, [a + b for a, b in zip(correctPerFrame, incorrectPerFrame)])
plt.legend(["Correct", "Incorrect", "Total"])
plt.xlabel("Time (hours)")
plt.ylabel("Number of Active Tracks")

plt.subplot(2, 2, 2)
plt.plot(frames, percentCorrect)
plt.xlabel("Time (hours)")
plt.ylabel("Correct Tracks (Fraction of Total)")
plt.axhline(y=min(percentCorrect), color="r", linestyle="dashed")
plt.text(0, min(percentCorrect), "%.2f" % min(percentCorrect), verticalalignment='bottom',
         horizontalalignment='left', color="r")
plt.ylim([0, 1.1])

idsToHighlight = [0, 1, 2, 3, 5, 6, 9, 18, 33]
plt.subplot(2, 2, 3)
plt.plot(frames, np.delete(areasGT, idsToHighlight, 1), 'o-', color=(0.9, 0.9, 0.9), label="_nolabel")
[plt.plot(frames, areasGT[:, idToHighlight], 'o-', label=str(idToHighlight)) for idToHighlight in idsToHighlight]
plt.legend()
plt.title("Ground Truth Areas")
plt.xlabel("Time (hours)")
plt.ylabel(r"Organoid Area (x $10^3 \mu m^2$)")

plt.subplot(2, 2, 4)
plt.title("Automated Areas")
plt.plot(frames, np.delete(areasAutomated, idsToHighlight, 1), 'o-', color=(0.9, 0.9, 0.9), label="_nolabel")
[plt.plot(frames, areasAutomated[:, idToHighlight], 'o-', label=str(idToHighlight)) for idToHighlight in idsToHighlight]
plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel(r"Organoid Area (x $10^3 \mu m^2$)")

print(percentCorrect)

plt.show()
