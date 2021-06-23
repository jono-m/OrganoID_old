from SettingsParser import JobSettings

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from DataGenerator import DataGenerator

from sklearn.model_selection import train_test_split

from pathlib import Path
import typing

import dill
from realTimeData import RealTimeData


class RealTimeCallback(Callback):
    def __init__(self, path: Path, totalBatches: int):
        super().__init__()
        self.path = path

        self.totalBatches = totalBatches
        self.data = RealTimeData(totalBatches)

    def on_train_batch_end(self, batch, logs=None):
        if not logs or 'loss' not in logs:
            return
        current = self.data.epochs[-1]
        current.losses.append(logs['loss'])
        current.accuracies.append(logs['accuracy'])
        current.meanIOUs.append(logs['MeanIOU'])
        self.Rewrite()

    def on_epoch_begin(self, epoch, logs=None):
        self.data.epochs.append(RealTimeData.EpochData())
        self.Rewrite()

    def Rewrite(self):
        with open(self.path, "wb") as file:
            dill.dump(self.data, file)


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def __init__(self):
        super().__init__(num_classes=2, name="MeanIOU")

    def update_state(self, y_true, y_pred, sample_weight=None):
        formattedTrue = tf.cast(y_true, tf.uint8)
        formattedPredictions = tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.uint8),
        return super().update_state(formattedTrue, formattedPredictions, sample_weight)


def DoTraining(settings: JobSettings):
    FitModel(settings.ImagesPath(), settings.SegmentationsPath(), settings.OutputPath(),
             epochs=settings.Epochs(), test_size=settings.GetTestSplit(), patience=5,
             batch_size=settings.GetBatchSize(), imageSize=settings.GetSize(), dropout_rate=settings.GetDropoutRate(),
             numImages=settings.GetImageNumber())


def FitModel(trainingImagesPath: Path, trainingSegmentationsPath: Path, outputPath: Path, epochs: int,
             test_size=0.5, batch_size=1, patience=5, imageSize: typing.Tuple[int, int] = (640, 640), numImages=-1,
             dropout_rate=0.1, learning_rate=0.00001):
    print("-----------------------")
    print("Building model...")
    print("\tImages directory: " + str(trainingImagesPath))
    print("\tSegmentations directory: " + str(trainingSegmentationsPath))
    print("\tModel directory: " + str(outputPath))

    imagePaths = [imagePath for imagePath in sorted(trainingImagesPath.iterdir()) if imagePath.is_file()]
    imagePaths = imagePaths[:numImages]
    segmentationPaths = [segmentationPath for segmentationPath in sorted(trainingSegmentationsPath.iterdir())
                         if segmentationPath.is_file()]
    segmentationPaths = segmentationPaths[:numImages]

    print("\tDone.")

    print("\tSplitting training and testing datasets (" + str(test_size * 100) + "% for testing)...")
    if test_size == 0:
        trainingImagePaths, testingImagePaths, trainingSegmentationPaths, testingSegmentationPaths = \
            imagePaths, imagePaths, segmentationPaths, segmentationPaths
    else:
        trainingImagePaths, testingImagePaths, trainingSegmentationPaths, testingSegmentationPaths = train_test_split(
            imagePaths,
            segmentationPaths,
            test_size=test_size)

    print("\tDone!")

    print("\tBuilding model pipeline...")
    inputs = Input((imageSize[0], imageSize[1], 1))

    startSize = 64

    currentLayer = inputs
    contractingLayers = []
    for i in range(5):
        if i > 0:
            currentLayer = MaxPooling2D((2, 2))(currentLayer)

        size = startSize * 2 ** i
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)
        currentLayer = Dropout(dropout_rate)(currentLayer)
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)
        contractingLayers.append(currentLayer)

    for i in reversed(range(4)):
        size = startSize * 2 ** i

        currentLayer = Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same')(currentLayer)
        currentLayer = concatenate([currentLayer, contractingLayers[i]])
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)
        currentLayer = Dropout(dropout_rate)(currentLayer)
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)

    final = Conv2D(1, (1, 1), activation='sigmoid')(currentLayer)

    model = Model(inputs=[inputs], outputs=[final])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-4),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[MyMeanIOU()])
    model.summary()
    print("\tRequired memory: " + str(keras_model_memory_usage_in_bytes(model, batch_size=batch_size)))
    print("\tDone!")

    earlystopper = EarlyStopping(patience=patience, verbose=1)
    outputPath.mkdir(parents=True, exist_ok=True)
    print("\tFitting model...", flush=True)
    model.fit(DataGenerator(trainingImagePaths, trainingSegmentationPaths, imageSize, batch_size),
              validation_data=DataGenerator(testingImagePaths, testingSegmentationPaths, imageSize, batch_size),
              verbose=1,
              epochs=epochs,
              callbacks=[earlystopper])
    print("\tDone!", flush=True)

    print("\tSaving model...")

    modelJobSavePath = outputPath / "trainedModel"

    model.save(str(modelJobSavePath.absolute()))

    results = modelJobSavePath
    print("Model saved to " + str(results))

    return results


def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
            batch_size * shapes_mem_count
            + internal_model_mem_count
            + trainable_count
            + non_trainable_count
    )
    return total_memory
