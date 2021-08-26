from typing import List
import numpy as np
from skimage.measure import regionprops, label


class Analyzer:
    class TimePointData:
        def __init__(self, organoidAreas: List[int]):
            self.organoidAreas = organoidAreas

    def __init__(self):
        self.timePoints: List[Analyzer.TimePointData] = []

    def Reset(self):
        self.timePoints = []

    def AnalyzeImage(self, image: np.ndarray):
        regionInfo = regionprops(label(image))

        areas = [region.area for region in regionInfo]
        self.timePoints.append(Analyzer.TimePointData(areas))
