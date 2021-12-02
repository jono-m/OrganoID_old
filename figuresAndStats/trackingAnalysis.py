import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

import matplotlib.pyplot as plt
from skimage.measure import regionprops
from backend.ImageManager import LoadImages, LabelTracks, SaveGIF
from backend.Tracker import Tracker
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


automatedLabelMaps = ParseMap(r"dataset\demo\tracked\trackResults.csv")
groundTruthLabelMaps = ParseMap(r"dataset\demo\tracked\groundTruth\mapping.csv")
labeledImages = \
    list(LoadImages(r"dataset\demo\labeled\20210127_organoidplate003_XY36_Z3_C2_detected_labeled.tiff"))[
        0].frames

regionPropsAll = [regionprops(image) for image in labeledImages]

automatedTracks = BuildTracks(regionPropsAll, automatedLabelMaps)
groundTruthTracks = BuildTracks(regionPropsAll, groundTruthLabelMaps)


def GetKeyForTrack(track: Tracker.OrganoidTrack):
    return track.data[0].centroid[0] * 100000 + track.data[0].centroid[1] * 1000 + track.firstFrame


automatedTracks.sort(key=GetKeyForTrack)
groundTruthTracks.sort(key=GetKeyForTrack)

originalImage = list(LoadImages(
    r"dataset\demo\20210127_organoidplate003_XY36_Z3_C2.tif", (512, 512), "L"))[0].frames

maxNumber = max(len(automatedTracks), len(groundTruthTracks))
for i in range(maxNumber):
    if i < len(automatedTracks):
        automatedTracks[i].id = i
    if i < len(groundTruthTracks):
        groundTruthTracks[i].id = i

renumberedAutomatedImages = LabelTracks(automatedTracks, (255, 255, 255), 255, 100, (0, 255, 0), (255, 0, 0),
                                        originalImage)
renumberedGTImages = LabelTracks(groundTruthTracks, (255, 255, 255), 255, 100, (0, 255, 0), (255, 0, 0),
                                 originalImage)

merged = [np.concatenate([a, b], axis=1) for a, b in zip(renumberedGTImages, renumberedAutomatedImages)]

SaveGIF(merged, Path(r"figuresAndStats\renumberedTracks.gif"))
