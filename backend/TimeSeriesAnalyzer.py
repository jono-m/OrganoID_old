from typing import List
import numpy as np
from skimage.measure import regionprops, label


class TimeSeriesAnalyzer:
    class TimePointData:
        def __init__(self, organoidAreas: List[int]):
            self.organoidAreas = organoidAreas

    def __init__(self):
        self.timePoints: List[TimeSeriesAnalyzer.TimePointData] = []

    def Reset(self):
        self.timePoints = []

    def AnalyzeImage(self, image: np.ndarray):
        regionInfo = regionprops(label(image))

        areas = [region.area for region in regionInfo]
        self.timePoints.append(TimeSeriesAnalyzer.TimePointData(areas))
