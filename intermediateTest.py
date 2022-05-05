from pathlib import Path

import keras
import numpy as np
from PIL import Image
from backend.ImageManager import LoadImages
import tensorflow as tf


def Contrast(i):
    return 255 * (i - i.min()) / (i.max() - i.min())


def Visualize(i, size=None):
    i = i[0, :, :, :]
    blockHeight, blockWidth, numBlocks = i.shape
    rows = 2 ** int(np.log2(np.sqrt(numBlocks)))
    columns = int(numBlocks / rows)
    image = np.zeros([rows * blockHeight, columns * blockWidth], dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            blockNumber = row * columns + column
            block = i[:, :, blockNumber]
            image[(row * blockHeight):((row + 1) * blockHeight),
            (column * blockWidth):((column + 1) * blockWidth)] = Contrast(block)
    image = Image.fromarray(image)
    if(size):
        image = image.resize([image.size[0]*size, image.size[1]*size])
    Image.fromarray(image).show()

image = LoadImages(Path(r"dataset\demo"), mode="L")
image = list(image)[0].frames[0]
image = np.asarray(Image.fromarray(image).resize([512, 512]))
image = np.reshape(image, [1, 512, 512, 1]).astype(np.float32)
image = Contrast(image)

model = tf.keras.models.load_model(str(Path(r"model2\fullModel")))

Visualize(model.get_layer("conv2d").get_weights()[0].reshape(1, 3, 3, 8), 8)
#
# layers = ["conv2d_18"]
# for layer in layers:
#     intermediateModel = keras.Model(inputs=model.input, outputs=model.get_layer(layer).output)
#     intermediateOutput = intermediateModel.predict(image)
#     Visualize(intermediateOutput)