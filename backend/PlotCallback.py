from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from pathlib import Path


class PlotCallback(Callback):
    def __init__(self, outPath: Path):
        super().__init__()
        self.i = 0
        self.path = outPath
        self.path.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        model = self.model

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        liteModel = converter.convert()
        savePath = self.path / ("epoch_" + str(epoch) + ".tflite")
        with open(savePath, "wb") as f:
            f.write(liteModel)
