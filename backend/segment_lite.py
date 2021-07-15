import tflite_runtime.interpreter as tflite
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage


class SmartInterpreter:
    def __init__(self, modelPath: Path):
        self._interpreter = tflite.Interpreter(model_path=str(modelPath.absolute()))
        self._inputIndex = self._interpreter.get_input_details()[0]['index']
        self._inputShape = self._interpreter.get_input_details()[0]['shape']
        self._output_index = self._interpreter.get_output_details()[0]['index']
        self.inputShape = self._inputShape[1:3]
        self._interpreter.allocate_tensors()

    def Predict(self, image: np.ndarray):
        image = np.reshape(np.array(image), self._inputShape).astype(np.float32)
        self._interpreter.set_tensor(self._inputIndex, image)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)
        return output[0, :, :, 0]


def PostSegment(image: np.ndarray, threshold: float, opening):
    image = image > threshold
    if opening > 0:
        image = ndimage.binary_opening(image, iterations=opening)
    return image
