# Tracker.py -- tracks organoids in sequences of labeled images

from typing import List
import numpy as np
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
from util import Printer


class Tracker:
    # Data point for one frame for one organoid
    class OrganoidFrameData:
        def __init__(self, rp):
            self.regionProperties = rp
            self.wasDetected = True

        def Duplicate(self):
            # Clone this data point
            return Tracker.OrganoidFrameData(self.regionProperties)

    class OrganoidTrack:
        # Collection of data points for a single identified organoid

        def __init__(self, frame, newID):
            self.id = newID

            self.active = True
            self.firstFrame = frame
            self.age = 0
            self.invisibleConsecutive = 0
            self.data: List[Tracker.OrganoidFrameData] = []

        def LastDetectionFrame(self):
            for frameNumber, data in reversed(list(enumerate(self.data))):
                if data.wasDetected:
                    return frameNumber + self.firstFrame
            return self.firstFrame

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

        def Detect(self, rp):
            # Report a detection of this track at this frame
            self.data.append(Tracker.OrganoidFrameData(rp))
            self.invisibleConsecutive = 0
            self.age += 1

    def __init__(self):
        self._tracks: List[Tracker.OrganoidTrack] = []
        self.overlapCost = 100
        self.costOfNewOrganoid = 1
        self.costOfMissingOrganoid = 1
        self.deleteTracksAfterMissing = -1
        self.frame = 0
        self.nextID = 0

    def Track(self, image: np.ndarray):
        # Morphologically analyze labled regions in the image
        detections = regionprops(image)
        coordinates = [detection.coords for detection in detections]

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

        print("Computing cost matrix... ")
        # Fill in the cost of assignment for each track to each detection
        for trackNumber in range(numTracks):
            Printer.printRep("(Track %d/%d)" % (trackNumber, numTracks))

            overlapCosts = self.OverlapCost(availableTracks[trackNumber].LastData().regionProperties.coords,
                                            coordinates)
            costMatrix[trackNumber, 0:numDetections] = overlapCosts

        Printer.printRep(None)
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
                track = Tracker.OrganoidTrack(self.frame, self.nextID)
                self.nextID += 1
                self._tracks.append(track)
            else:
                # This got assigned to an existing track.
                track = availableTracks[trackIndex]

            # Register the detection.
            track.Detect(detections[detectionIndex])

        # Go through all tracks and inactivate any that have been missing for more than a given number of frames.
        if self.deleteTracksAfterMissing >= 0:
            for track in availableTracks:
                if track.invisibleConsecutive >= self.deleteTracksAfterMissing:
                    track.active = False

        self.frame += 1

    def GetTracks(self):
        return self._tracks

    def OverlapCost(self, coordinatesA, bCoordinates):
        overlaps = []
        for coordinatesB in bCoordinates:
            overlap = np.count_nonzero((coordinatesA[:, None] == coordinatesB).all(-1).any(-1))
            if overlap == 0:
                overlap = np.inf
            else:
                overlap = 1 / overlap
            overlaps.append(overlap * self.overlapCost)
        return np.asarray(overlaps)
