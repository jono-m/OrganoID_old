import matplotlib.pyplot as plt
from SettingsParser import JobSettings
import dill
from realTimeData import RealTimeData
from pathlib import Path


def DoMonitor(settings: JobSettings):
    plotter = Plotter(settings.GetLogPath())
    while True:
        plotter.Tick()


class Plotter:
    def __init__(self, path: Path):
        self.path = path

        self._lastLoadTime = -1
        self.data = self.Reload()

        plt.ion()

        self.lossPlot = plt.subplot(1, 3, 1)

        self.accuracyPlot = plt.subplot(1, 3, 2)

        self.iouPlot = plt.subplot(1, 3, 2)

        self.Redraw()

    def Tick(self):
        if self.path.stat().st_mtime > self._lastLoadTime:
            self.data = self.Reload()
            self.Redraw()

    def Redraw(self):
        plots = [self.lossPlot, self.accuracyPlot, self.iouPlot]
        [plot.clear() for plot in plots]

        for epoch in self.data.epochs:
            points = len(epoch.losses)
            if epoch == self.data.epochs[-1]:
                width = 2
                alpha = 1
            else:
                width = 1
                alpha = 0.5

            [plot.plot(points, [epoch.losses, epoch.accuracies, epoch.meanIOUs][i], linewidth=width, alpha=alpha) for
             (i, plot) in enumerate(plots)]

        [plot.set_xlim(0, self.data.batchesPerEpoch) for plot in plots]
        [plot.set_xlabel("Batch Number") for plot in plots]
        [plot.set_title(["Loss", "Accuracy", "Mean IOU"][i]) for (i, plot) in enumerate(plots)]
        [plot.set_ylim(0, plot.get_ylim[1]) for plot in plots]

    def Reload(self):
        with open(self.path, "rb") as file:
            data: RealTimeData = dill.load(file)
        self._lastLoadTime = self.path.stat().st_mtime
        return data
