try:
    from tflite_runtime.interpreter import Interpreter
except ImportError as e:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
from pathlib import Path
from backend.ImageManager import ContrastOp
import numpy as np
import typing


class Segmenter:
    def __init__(self, modelPath: Path):
        self._interpreter = Interpreter(model_path=str(modelPath.absolute()))
        self._inputIndex = self._interpreter.get_input_details()[0]['index']
        self._inputShape = self._interpreter.get_input_details()[0]['shape']
        self._output_index = self._interpreter.get_output_details()[0]['index']
        self._interpreter.allocate_tensors()

    def Segment(self, image: np.ndarray) -> np.ndarray:
        if True:
            image = 255 * ((image - image.min()) / (image.max() - image.min()))
        image = np.reshape(image, self._inputShape).astype(np.float32)
        self._interpreter.set_tensor(self._inputIndex, image)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)
        return output[0, :, :, 0]

    def SegmentMultiple(self, images: typing.List[np.ndarray]) -> np.ndarray:
        stackedImages = np.expand_dims(np.stack(images), axis=-1).astype(np.float32)
        batchShape = self._inputShape
        batchShape[0] = len(images)
        self._interpreter.resize_tensor_input(self._inputIndex, batchShape, True)
        self._interpreter.allocate_tensors()
        self._interpreter.set_tensor(self._inputIndex, stackedImages)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)
        return output[:, :, :, 0]
