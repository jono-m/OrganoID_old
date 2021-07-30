import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, \
    BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from pathlib import Path
from backend.ModelDataGenerator import ModelDataGenerator


class ModelTrainer:
    def __init__(self, inputSize, dropoutRate, startSize):
        inputs = Input((inputSize[0], inputSize[1], 1))

        currentLayer = inputs
        currentLayer = BatchNormalization()(currentLayer)
        contractingLayers = []
        for i in range(5):
            if i > 0:
                currentLayer = MaxPooling2D((2, 2))(currentLayer)

            currentLayer = BatchNormalization()(currentLayer)
            size = startSize * 2 ** i
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)
            currentLayer = BatchNormalization()(currentLayer)
            currentLayer = Dropout(dropoutRate * i)(currentLayer)
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)
            currentLayer = BatchNormalization()(currentLayer)
            contractingLayers.append(currentLayer)

        for i in reversed(range(4)):
            size = startSize * 2 ** i

            currentLayer = Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same')(currentLayer)
            currentLayer = concatenate([currentLayer, contractingLayers[i]])
            currentLayer = BatchNormalization()(currentLayer)
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)
            currentLayer = BatchNormalization()(currentLayer)
            currentLayer = Dropout(dropoutRate * i)(currentLayer)
            currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
                currentLayer)
            currentLayer = BatchNormalization()(currentLayer)

        final = Conv2D(1, (1, 1), activation='sigmoid')(currentLayer)

        self._model = Model(inputs=[inputs], outputs=[final])

    def Train(self, learningRate, patience, epochs, trainingDataGenerator: ModelDataGenerator,
              testingDataGenerator: ModelDataGenerator,
              outputPath: Path = None):
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                            loss=tf.keras.losses.binary_crossentropy)

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

    @staticmethod
    def ConvertToLiteModel(fullModel):
        converter = tf.lite.TFLiteConverter.from_keras_model(fullModel)
        return converter.convert()

    @staticmethod
    def SaveLiteModel(path: Path, liteModel):
        savePath = path
        with open(savePath, "wb") as f:
            f.write(liteModel)

    class ModelSavingCallback(Callback):
        def __init__(self, outPath: Path):
            super().__init__()
            self.i = 0
            self.path = outPath

        def on_epoch_end(self, epoch, logs=None):
            savePath = self.path / ("epoch_" + str(epoch) + ".tflite")

            ModelTrainer.SaveLiteModel(savePath, ModelTrainer.ConvertToLiteModel(self.model))
