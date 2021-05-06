import typing


class RealTimeData:
    class EpochData:
        def __init__(self):
            self.losses = []
            self.accuracies = []
            self.meanIOUs = []

    def __init__(self, batches: int):
        self.batchesPerEpoch = batches
        self.epochs: typing.List[RealTimeData.EpochData] = []
