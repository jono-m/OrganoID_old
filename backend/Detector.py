from tflite_runtime.interpreter import Interpreter

from pathlib import Path
from PIL import Image
import numpy as np
import skimage.color as colors
import typing


class Detector:
    def __init__(self, modelPath: Path):
        self._interpreter = Interpreter(model_path=str(modelPath.absolute()))
        self._inputIndex = self._interpreter.get_input_details()[0]['index']
        self._inputShape = self._interpreter.get_input_details()[0]['shape']
        self._output_index = self._interpreter.get_output_details()[0]['index']
        self._interpreter.allocate_tensors()

    def Detect(self, image: np.ndarray) -> np.ndarray:
        inSize = reversed(image.shape[:2])
        image = np.asarray(Image.fromarray(image).resize([512, 512]))
        image = 255 * ((image - image.min()) / (image.max() - image.min()))
        image = np.reshape(image, self._inputShape).astype(np.float32)
        self._interpreter.set_tensor(self._inputIndex, image)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)[0, :, :, 0]
        output = np.asarray(Image.fromarray(output).resize(inSize))
        return output

    def DetectMultiple(self, images: typing.List[np.ndarray]) -> np.ndarray:
        stackedImages = np.expand_dims(np.stack(images), axis=-1).astype(np.float32)
        batchShape = self._inputShape
        batchShape[0] = len(images)
        self._interpreter.resize_tensor_input(self._inputIndex, batchShape, True)
        self._interpreter.allocate_tensors()
        self._interpreter.set_tensor(self._inputIndex, stackedImages)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)
        return output[:, :, :, 0]

    @staticmethod
    def ConvertToHeatmap(detected: np.ndarray) -> np.ndarray:
        minimum = detected.min()
        maximum = detected.max()
        hue = 44.8 / 360
        h = np.ones_like(detected) * hue
        s = np.minimum(1, 2 - 2 * (detected - minimum) / (maximum - minimum))
        v = np.minimum(1, 2 * (detected - minimum) / (maximum - minimum))
        concat = np.stack([h, s, v], -1)
        converted = colors.hsv2rgb(concat)
        return (converted * 255).astype(np.uint8)
