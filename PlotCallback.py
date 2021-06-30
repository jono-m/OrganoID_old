from tensorflow.keras.callbacks import Callback
from segmentation import SegmentImage, Image
from pathlib import Path
import matplotlib.pyplot as plt


class PlotCallback(Callback):
    def __init__(self, testPath, outPath: Path):
        super().__init__()
        self.i = 0
        self.path = outPath
        self.path.mkdir(parents=True, exist_ok=True)
        self.image = Image.open(testPath)

        self.trainLosses = []
        self.trainIOUS = []
        self.testLosses = []
        self.testIOUS = []

    def on_epoch_end(self, epoch, logs=None):
        segmented = SegmentImage(self.image, self.model)
        segmented.save(self.path / ("image_" + str(self.i) + ".png"))
        self.i += 1

        self.trainLosses.append(logs['loss'])
        self.trainIOUS.append(logs['MeanIOU'])
        self.testLosses.append(logs['val_loss'])
        self.testIOUS.append(logs['val_MeanIOU'])

    def on_train_end(self, logs=None):
        timePoints = range(len(self.trainIOUS))
        plt.subplot(1, 2, 1)
        plt.plot(timePoints, self.trainLosses, 'bo-')
        plt.plot(timePoints, self.testLosses, 'ro-')
        plt.title("Binary Cross-Entropy")
        plt.legend(["Training", "Validation"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        plt.title("Mean Jaccard Index")
        plt.plot(timePoints, self.trainIOUS, 'bo-')
        plt.plot(timePoints, self.testIOUS, 'ro-')
        plt.legend(["Training", "Validation"])
        plt.xlabel("Epoch")
        plt.ylabel("Mean IOU")

        plt.savefig(self.path / "finalPerformance.png")
