# ModelTrainer.py -- uses Keras/TensorFlow to train a u-net convoluational neural network.

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from pathlib import Path
from backend.ModelDataGenerator import ModelDataGenerator


class ModelTrainer:
    def __init__(self, inputSize, dropoutRate, startSize):
        # First layer in the network is each pixel in the grayscale image
        inputs = Input((inputSize[0], inputSize[1], 1))

        # Contracting path identifies features at increasing levels of detail
        # (i.e. intensity, edges, shapes, texture...)
        currentLayer = inputs
        contractingLayers = []
        for i in range(5):
            if i > 0:
                currentLayer = MaxPooling2D((2, 2))(currentLayer)

            size = startSize * 2 ** i
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)
            currentLayer = Dropout(dropoutRate * i)(currentLayer)
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)
            contractingLayers.append(currentLayer)

        # Expanding path spatially places recognized features.
        for i in reversed(range(4)):
            size = startSize * 2 ** i

            currentLayer = Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same')(currentLayer)
            currentLayer = concatenate([currentLayer, contractingLayers[i]])
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)
            currentLayer = Dropout(dropoutRate * i)(currentLayer)
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)

        # Last layer is sigmoid to produce probability map.
        final = Conv2D(1, (1, 1), activation='sigmoid')(currentLayer)

        self._model = Model(inputs=[inputs], outputs=[final])
        self._model.summary()

    def Train(self, learningRate, patience, epochs, trainingDataGenerator: ModelDataGenerator,
              testingDataGenerator: ModelDataGenerator,
              outputPath: Path = None):
        # Adam optimizer is used for SGD. Binary cross-entropy for loss.
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                            loss=tf.keras.losses.binary_crossentropy)

        # Stop training once performance on validation dataset has reached a local minimum (window size is "patience").
        callbacks = [EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)]
        if outputPath is not None:
            subPath = outputPath / "epochs"
            subPath.mkdir(parents=True, exist_ok=True)
            callbacks.append(ModelTrainer.ModelSavingCallback(subPath))

        self._model.fit(trainingDataGenerator,
                        validation_data=testingDataGenerator,
                        verbose=1,
                        epochs=epochs,
                        callbacks=callbacks)

        if outputPath is not None:
            self.SaveLiteModel(outputPath / "model.tflite", self.ConvertToLiteModel(self._model))
            self.SaveModel(outputPath / "fullModel", self._model)

    # Use TFLite to minimize memory overhead for saved models and inference.
    @staticmethod
    def ConvertToLiteModel(fullModel):
        converter = tf.lite.TFLiteConverter.from_keras_model(fullModel)
        return converter.convert()

    @staticmethod
    def SaveLiteModel(path: Path, liteModel):
        savePath = path
        with open(savePath, "wb") as f:
            f.write(liteModel)

    @staticmethod
    def SaveModel(path: Path, fullModel: Model):
        fullModel.save(path)

    # After every epoch, save the training and validation performance, as well as a copy of the model.
    class ModelSavingCallback(Callback):
        def __init__(self, outPath: Path):
            super().__init__()
            self.i = 0
            self.path = outPath
            self.trainLosses = []
            self.validationLosses = []

        def on_epoch_end(self, epoch, logs=None):
            savePath = self.path / ("epoch_" + str(epoch) + ".tflite")

            ModelTrainer.SaveLiteModel(savePath, ModelTrainer.ConvertToLiteModel(self.model))
            ModelTrainer.SaveModel(self.path / ("epoch_" + str(epoch) + "_fullModel"), self.model)
            self.trainLosses.append(str(logs['loss']))
            self.validationLosses.append(str(logs['val_loss']))

        def on_train_end(self, logs=None):
            outfile = open(self.path / "losses.csv", "w+")
            outfile.write("Training loss, " + ",".join(self.trainLosses) + "\n")
            outfile.write("Validation loss, " + ",".join(self.validationLosses) + "\n")
            outfile.close()
