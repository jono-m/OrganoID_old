import keras.saving.saved_model.model_serialization
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

    def ListIntermediates(self):
        return [x for x in self._interpreter.get_tensor_details() if len(x['shape']) == 4 and len(x['name']) < 35]

    def Visualize(self, i):
        i = i[0, :, :, :]
        blockHeight, blockWidth, numBlocks = i.shape
        rows = 2**int(np.log2(np.sqrt(numBlocks)))
        columns = int(numBlocks/rows)
        image = np.zeros([rows*blockHeight, columns*blockWidth], dtype=np.float32)
        for row in range(rows):
            for column in range(columns):
                blockNumber = row*columns + column
                block = i[:, :, blockNumber]
                image[(row*blockHeight):((row+1)*blockHeight),(column*blockWidth):((column+1)*blockWidth)] = self.Contrast(block)
        Image.fromarray(image).show()

    def GetIntermediateByIndex(self, i):
        return self.GetIntermediate(self.ListIntermediates()[i]['name'])

    def Contrast(self, i):
        return 255 * (i - i.min()) / (i.max()-i.min())

    def GetIntermediate(self, layerName):
        eluTargetName = "model/" + layerName + "/Elu"
        maxPoolTargetName = "model/" + layerName + "/MaxPool"
        concatTargetName = "model/" + layerName + "/concat"
        names = [eluTargetName, maxPoolTargetName, concatTargetName, layerName]
        index = [i for i, tensorDetail in enumerate(self._interpreter.get_tensor_details()) if tensorDetail['name'] in names]
        print("Index: " + str(index))
        return self._interpreter.get_tensor(index[0])

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
