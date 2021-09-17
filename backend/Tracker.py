from typing import List
import numpy as np
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment


class Tracker:
    class TrackData:
        def __init__(self, centroid, area, pixels, detection, image, bbox):
            self.centroid = centroid
            self.area = area
            self.pixels = pixels
            self.detection = detection
            self.image = image
            self.bbox = bbox

        def Duplicate(self):
            return Tracker.TrackData(self.centroid, self.area, self.pixels, self.detection, self.image, self.bbox)

    class OrganoidTrack:
        nextID = 1

        def __init__(self, frame, centroid, area, pixels, image, bbox):
            self.id = Tracker.OrganoidTrack.nextID
            Tracker.OrganoidTrack.nextID += 1
            self.data = [Tracker.TrackData(centroid, area, pixels, True, image, bbox)]
            self.active = True
            self.firstFrame = frame
            self.age = 0
            self.invisibleConsecutive = 0

        def DataAtFrame(self, frame):
            if frame < self.firstFrame:
                return None
            else:
                local = frame - self.firstFrame
                if local >= len(self.data):
                    return None
                else:
                    return self.data[local]

        def NoDetection(self):
            data = self.data[-1].Duplicate()
            data.detection = False
            self.data.append(data)
            self.invisibleConsecutive += 1
            self.age += 1

        def LastData(self):
            return self.data[-1]

        def Detect(self, centroid, area, pixels, image, bbox):
            self.data.append(Tracker.TrackData(centroid, area, pixels, True, image, bbox))
            self.invisibleConsecutive = 0
            self.age += 1

    def __init__(self):
        self._tracks: List[Tracker.OrganoidTrack] = []
        self.distanceCost = 1
        self.areaCost = 0
        self.costOfNewOrganoid = 100
        self.costOfMissingOrganoid = 20
        self.deleteTracksAfterMissing = 10
        self.frame = 0

    def Track(self, image: np.ndarray):
        detections = regionprops(image)
        centroids = np.array([detection.centroid for detection in detections])
        coordinates = [detection.coords for detection in detections]
        images = [detection.image for detection in detections]
        bboxes = [detection.bbox for detection in detections]
        areas = np.array([detection.area for detection in detections])
        availableTracks = [track for track in self._tracks if track.active]
        numTracks = len(availableTracks)
        numDetections = len(detections)

        fullSize = numTracks + numDetections
        costMatrix = np.zeros([fullSize, fullSize])

        newOrganoidMatrix = np.full([numDetections, numDetections], np.inf)
        np.fill_diagonal(newOrganoidMatrix, self.costOfNewOrganoid)

        missingOrganoidMatrix = np.full([numTracks, numTracks], np.inf)
        np.fill_diagonal(missingOrganoidMatrix, self.costOfMissingOrganoid)

        costMatrix[numTracks:, :numDetections] = newOrganoidMatrix
        costMatrix[:numTracks, numDetections:] = missingOrganoidMatrix

        for trackNumber in range(numTracks):
            distanceCosts = self.DistanceCost(availableTracks[trackNumber].LastData().centroid, centroids)
            areaCosts = self.AreaCost(availableTracks[trackNumber].LastData().area, areas)
            costMatrix[trackNumber, 0:numDetections] = distanceCosts + areaCosts

        trackIndices, detectionIndices = linear_sum_assignment(costMatrix)
        for assignmentIndex in range(trackIndices.size):
            trackIndex = trackIndices[assignmentIndex]
            detectionIndex = detectionIndices[assignmentIndex]
            if detectionIndex >= numDetections:
                # Track didn't get assigned.
                if trackIndex >= numTracks:
                    # It's a dummy track.
                    continue
                else:
                    # The track lost its target for this frame.
                    availableTracks[trackIndex].NoDetection()
                    continue

            area = areas[detectionIndex]
            centroid = centroids[detectionIndex]
            image = images[detectionIndex]
            bbox = bboxes[detectionIndex]
            objectCoordinates = coordinates[detectionIndex]
            if trackIndex >= numTracks:
                # This got assigned to a new track!
                matchedTrack = Tracker.OrganoidTrack(self.frame, centroid, area, objectCoordinates, image, bbox)
                self._tracks.append(matchedTrack)
            else:
                # This got assigned to an existing track.
                availableTracks[trackIndex].Detect(centroid, area, objectCoordinates, image, bbox)

        for track in availableTracks:
            if track.invisibleConsecutive >= self.deleteTracksAfterMissing:
                track.active = False

        self.frame += 1

    def GetTracks(self):
        return self._tracks

    def DistanceCost(self, centroidA, bCentroids):
        distances = np.sqrt(np.sum(np.square(bCentroids - centroidA), axis=1))
        return distances * self.distanceCost

    def AreaCost(self, areaA, bAreas):
        return abs(areaA - bAreas) * self.areaCost
