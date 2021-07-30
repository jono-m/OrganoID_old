from typing import List
import numpy as np
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment


class Tracker:
    class OrganoidTrack:
        nextID = 1

        def __init__(self, frame, centroid, area):
            self.id = Tracker.OrganoidTrack.nextID
            Tracker.OrganoidTrack.nextID += 1
            self.lastCentroid = centroid
            self.lastArea = area
            self.firstFrame = frame
            self.age = 0
            self.invisibleConsecutive = 0

        def NoDetection(self):
            self.invisibleConsecutive += 1
            self.age += 1

        def Detect(self, centroid, area):
            self.lastArea = area
            self.lastCentroid = centroid
            self.invisibleConsecutive = 0
            self.age += 1

    def __init__(self):
        self._trackedImages: List[np.ndarray] = []
        self._tracks: List[Tracker.OrganoidTrack] = []
        self.distanceCost = 1
        self.areaCost = 1
        self.costOfNonAssignment = 20
        self.deleteTracksAfterMissing = 3
        self.frame = 0

    def Track(self, image: np.ndarray):
        trackedImage = np.zeros_like(image)

        detections = regionprops(image)
        centroids = np.array([detection.centroid for detection in detections])
        areas = np.array([detection.area for detection in detections])
        labels = np.array([detection.label for detection in detections])
        numTracks = len(self._tracks)
        numDetections = len(detections)

        fullSize = max(numTracks, numDetections)
        costMatrix = np.ones([fullSize, fullSize]) * self.costOfNonAssignment

        for trackNumber in range(numTracks):
            distanceCosts = self.DistanceCost(self._tracks[trackNumber].lastCentroid, centroids)
            areaCosts = self.AreaCost(self._tracks[trackNumber].lastArea, areas)
            costMatrix[trackNumber, 0:numDetections] = distanceCosts + areaCosts

        trackIndices, detectionIndices = linear_sum_assignment(costMatrix)
        for assignmentIndex in range(trackIndices.size):
            trackIndex = trackIndices[assignmentIndex]
            detectionIndex = detectionIndices[assignmentIndex]
            if detectionIndex >= numDetections:
                # This track didn't get an assignment.
                self._tracks[trackIndex].NoDetection()
                continue

            area = areas[detectionIndex]
            centroid = centroids[detectionIndex]
            label = labels[detectionIndex]
            if trackIndex >= numTracks:
                matchedTrack = Tracker.OrganoidTrack(self.frame, centroid, area)
                self._tracks.append(matchedTrack)
            else:
                matchedTrack = self._tracks[trackIndex]
                self._tracks[trackIndex].Detect(centroid, area)
            trackedImage[image == label] = matchedTrack.id

        self._trackedImages.append(trackedImage)

        for track in self._tracks.copy():
            if track.invisibleConsecutive >= self.deleteTracksAfterMissing:
                self._tracks.remove(track)

        self.frame += 1

    def GetTrackedImages(self) -> List[np.ndarray]:
        return self._trackedImages

    def DistanceCost(self, centroidA, bCentroids):
        distances = np.sqrt(np.sum(np.square(bCentroids - centroidA), axis=1))
        return distances * self.distanceCost

    def AreaCost(self, areaA, bAreas):
        return abs(areaA - bAreas) * self.areaCost
