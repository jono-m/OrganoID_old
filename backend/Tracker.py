# Tracker.py -- tracks organoids in sequences of labeled images

from typing import List
import numpy as np
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment


class Tracker:
    # Data point for one frame for one organoid
    class OrganoidFrameData:
        def __init__(self, centroid, area, pixels, wasDetected, image, bbox, label):
            self.centroid = centroid
            self.area = area
            self.pixels = pixels
            self.wasDetected = wasDetected
            self.image = image
            self.bbox = bbox
            self.label = label

        def Duplicate(self):
            # Clone this data point
            return Tracker.OrganoidFrameData(self.centroid, self.area, self.pixels, self.wasDetected, self.image,
                                             self.bbox, self.label)

    class OrganoidTrack:
        # Collection of data points for a single identified organoid

        # Static counter to assign the next organoid ID
        nextID = 1

        def __init__(self, frame):
            self.id = Tracker.OrganoidTrack.nextID
            Tracker.OrganoidTrack.nextID += 1

            self.active = True
            self.firstFrame = frame
            self.age = 0
            self.invisibleConsecutive = 0
            self.data = []

        def DataAtFrame(self, frame):
            # Retrieve the datapoint for the given frame
            if frame < self.firstFrame:
                return None
            else:
                # Frame number relative to when this track started
                local = frame - self.firstFrame
                if local >= len(self.data):
                    return None
                else:
                    return self.data[local]

        def NoDetection(self):
            # Report no detection of this track for the current frame
            data = self.data[-1].Duplicate()
            data.wasDetected = False
            self.data.append(data)
            self.invisibleConsecutive += 1
            self.age += 1

        def LastData(self):
            return self.data[-1]

        def Detect(self, centroid, area, pixels, image, bbox, label):
            # Report a detection of this track at this frame
            self.data.append(Tracker.OrganoidFrameData(centroid, area, pixels, True, image, bbox, label))
            self.invisibleConsecutive = 0
            self.age += 1

    def __init__(self):
        self._tracks: List[Tracker.OrganoidTrack] = []
        self.distanceCost = 1
        self.areaCost = 2
        self.costOfNewOrganoid = 100
        self.costOfMissingOrganoid = 20
        self.deleteTracksAfterMissing = 10
        self.frame = 0

    def Track(self, image: np.ndarray):
        # Morphologically analyze labled regions in the image
        detections = regionprops(image)
        centroids = np.array([detection.centroid for detection in detections])
        coordinates = [detection.coords for detection in detections]
        images = [detection.image for detection in detections]
        bboxes = [detection.bbox for detection in detections]
        labels = [detection.label for detection in detections]
        areas = np.array([detection.area for detection in detections])

        # Get all currently active organoid tracks
        availableTracks = [track for track in self._tracks if track.active]
        numTracks = len(availableTracks)
        numDetections = len(detections)

        # Build cost matrix (larger size with "dummy" rows and columns allows for assignment of detections to new tracks
        # or of existing tracks to missing detections.
        fullSize = numTracks + numDetections
        costMatrix = np.zeros([fullSize, fullSize])

        # Fill in the cost of creating a new track
        newOrganoidMatrix = np.full([numDetections, numDetections], np.inf)
        np.fill_diagonal(newOrganoidMatrix, self.costOfNewOrganoid)
        costMatrix[numTracks:, :numDetections] = newOrganoidMatrix

        # Fill in the cost of considering an organoid as as missing
        missingOrganoidMatrix = np.full([numTracks, numTracks], np.inf)
        np.fill_diagonal(missingOrganoidMatrix, self.costOfMissingOrganoid)
        costMatrix[:numTracks, numDetections:] = missingOrganoidMatrix

        # Fill in the cost of assignment for each track to each detection
        for trackNumber in range(numTracks):
            distanceCosts = self.DistanceCost(availableTracks[trackNumber].LastData().centroid, centroids)
            areaCosts = self.AreaCost(availableTracks[trackNumber].LastData().area, areas)
            costMatrix[trackNumber, 0:numDetections] = distanceCosts + areaCosts

        # Solve the assignment problem (Hungarian algorithm)
        trackIndices, detectionIndices = linear_sum_assignment(costMatrix)

        # Handle assignments
        for assignmentIndex in range(trackIndices.size):
            trackIndex = trackIndices[assignmentIndex]
            detectionIndex = detectionIndices[assignmentIndex]

            if detectionIndex >= numDetections and trackIndex >= numTracks:
                # This is a dummy track and dummy detection. Can skip it.
                continue

            if detectionIndex >= numDetections:
                # Track didn't get assigned to a detection, so it lost its target for this frame.
                availableTracks[trackIndex].NoDetection()
                continue

            if trackIndex >= numTracks:
                # This organoid didn't get assigned to an existing track, so it must be new!
                track = Tracker.OrganoidTrack(self.frame)
                self._tracks.append(track)
            else:
                # This got assigned to an existing track.
                track = availableTracks[trackIndex]

            # Register the detection.
            track.Detect(centroids[detectionIndex],
                         areas[detectionIndex],
                         coordinates[detectionIndex],
                         images[detectionIndex],
                         bboxes[detectionIndex],
                         labels[detectionIndex])

        # Go through all tracks and inactivate any that have been missing for more than a given number of frames.
        if self.deleteTracksAfterMissing >= 0:
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
        return abs(np.sqrt(areaA) - np.sqrt(bAreas)) * self.areaCost
