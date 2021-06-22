from SettingsParser import JobSettings

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback

import numpy as np
from PIL import Image

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
    def __init__(self, threshold):
        super().__init__(num_classes=2, name="MeanIOU")
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        formattedTrue = tf.cast(y_true, tf.uint8)
        formattedPredictions = tf.cast(tf.math.greater_equal(y_pred, self.threshold), tf.uint8),
        return super().update_state(formattedTrue, formattedPredictions, sample_weight)


def DoTraining(settings: JobSettings):
    FitModel(settings.ImagesPath(), settings.SegmentationsPath(), settings.OutputPath(),
             epochs=settings.Epochs(), test_size=settings.GetTestSplit(), patience=5,
             batch_size=settings.GetBatchSize(), imageSize=settings.GetSize(), dropout_rate=settings.GetDropoutRate(),
             numImages=settings.GetImageNumber())


def LoadImages(filenames: typing.List[Path], imageSize: typing.Tuple[int, int]):
    images = [Image.open(imagePath) for imagePath in filenames]
    for imageIndex, image in enumerate(images):
        if image.mode == 'I':
            image = image.point(lambda x: x * (1 / 255))
        images[imageIndex] = np.array(image.resize(imageSize).convert(mode="L"))
    images = np.expand_dims(np.moveaxis(np.stack(images, axis=-1), -1, 0), -1)
    return images


def LoadSegmentations(filenames: typing.List[Path], imageSize: typing.Tuple[int, int]):
    segmentations = [Image.open(segmentationPath) for segmentationPath in filenames]
    for segmentationIndex, segmentation in enumerate(segmentations):
        segmentations[segmentationIndex] = np.array(segmentation.resize(imageSize).convert(mode="1"))

    segmentations = np.moveaxis(np.stack(segmentations, axis=-1), -1, 0).astype(int)
    return segmentations


def ImagesLoader(imageFileNames: typing.List[Path], segmentationFileNames: typing.List[Path],
                 imageSize: typing.Tuple[int, int], batchSize: int):
    L = len(imageFileNames)

    # this line is just to make the generator infinite, keras needs that
    while True:
        batchStart = 0
        batchEnd = batchSize

        while batchStart < L:
            limit = min(batchEnd, L)
            X = LoadImages(imageFileNames[batchStart:limit], imageSize)
            Y = LoadSegmentations(segmentationFileNames[batchStart:limit], imageSize)

            yield X, Y  # a tuple with two numpy arrays with batch_size samples

            batchStart += batchSize
            batchEnd += batchSize


def FitModel(trainingImagesPath: Path, trainingSegmentationsPath: Path, outputPath: Path, epochs: int,
             test_size=0.5, batch_size=1, patience=5, imageSize: typing.Tuple[int, int] = (640, 640), numImages=-1,
             dropout_rate=0.1, learning_rate=0.001):
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

    c1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(dropout_rate)(c1)
    c1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(dropout_rate)(c2)
    c2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)

    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(dropout_rate)(c3)
    c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)

    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(dropout_rate)(c4)
    c4 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(dropout_rate)(c5)
    c5 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(dropout_rate)(c6)
    c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(dropout_rate)(c7)
    c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(dropout_rate)(c8)
    c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(dropout_rate)(c9)
    c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    final = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[final])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    print("\tRequired memory: " + str(keras_model_memory_usage_in_bytes(model, batch_size=batch_size)))
    print("\tDone!")

    batches = int(len(trainingImagePaths) / batch_size)
    earlystopper = EarlyStopping(patience=patience, verbose=1)
    outputPath.mkdir(parents=True, exist_ok=True)
    print("\tFitting model...", flush=True)
    model.fit(ImagesLoader(trainingImagePaths, trainingSegmentationPaths, imageSize, batch_size),
              validation_data=ImagesLoader(testingImagePaths, testingSegmentationPaths, imageSize, batch_size),
              verbose=1,
              epochs=epochs,
              steps_per_epoch=batches,
              validation_steps=int(len(testingImagePaths) / batch_size),
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
