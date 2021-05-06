import matplotlib.pyplot as plt
from SettingsParser import JobSettings
import dill
from realTimeData import RealTimeData
from pathlib import Path
import time


def DoMonitor(settings: JobSettings):
    plotter = Plotter(settings.GetLogPath())
    while True:
        plotter.Tick()
        plt.pause(0.5)


class Plotter:
    def __init__(self, path: Path):
        self.path = path

        self._lastLoadTime = -1
        self.data = RealTimeData(1)
        self.Reload()

        plt.ion()

        self.lossPlot = plt.subplot(1, 3, 1)

        self.accuracyPlot = plt.subplot(1, 3, 2)

        self.iouPlot = plt.subplot(1, 3, 3)

        self.Redraw()

    def Tick(self):
        if self.path.stat().st_mtime > self._lastLoadTime:
            self.Reload()
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

            [plot.plot(range(points), [epoch.losses, epoch.accuracies, epoch.meanIOUs][i], linewidth=width, alpha=alpha)
             for
             (i, plot) in enumerate(plots)]

        [plot.set_xlim(0, self.data.batchesPerEpoch) for plot in plots]
        [plot.set_xlabel("Batch Number") for plot in plots]
        [plot.set_title(["Loss", "Accuracy", "Mean IOU"][i]) for (i, plot) in enumerate(plots)]
        self.lossPlot.set_ylim(0, self.lossPlot.get_ylim()[1])
        self.accuracyPlot.set_ylim(0, 1)
        self.iouPlot.set_ylim(0, 1)

    def Reload(self):
        try:
            with open(self.path, "rb") as file:
                self.data = dill.load(file)
        except Exception as e:
            print(e)
        self._lastLoadTime = self.path.stat().st_mtime
