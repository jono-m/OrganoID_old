import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from DataGenerator import DataGenerator

from pathlib import Path
from PlotCallback import PlotCallback
import typing


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def __init__(self):
        super().__init__(num_classes=2, name="MeanIOU")

    def update_state(self, y_true, y_pred, sample_weight=None):
        formattedTrue = tf.cast(y_true, tf.uint8)
        formattedPredictions = tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.uint8),
        return super().update_state(formattedTrue, formattedPredictions, sample_weight)


def CreateModel(dropoutRate, learningRate, imageSize):
    print("\tBuilding model pipeline...")
    inputs = Input((imageSize[0], imageSize[1], 1))

    startSize = 16

    currentLayer = inputs
    contractingLayers = []
    for i in range(5):
        if i > 0:
            currentLayer = MaxPooling2D((2, 2))(currentLayer)

        size = startSize * 2 ** i
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)
        currentLayer = Dropout(dropoutRate*i)(currentLayer)
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)
        contractingLayers.append(currentLayer)

    for i in reversed(range(4)):
        size = startSize * 2 ** i

        currentLayer = Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same')(currentLayer)
        currentLayer = concatenate([currentLayer, contractingLayers[i]])
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)
        currentLayer = Dropout(dropoutRate*i)(currentLayer)
        currentLayer = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(
            currentLayer)

    final = Conv2D(1, (1, 1), activation='sigmoid')(currentLayer)

    model = Model(inputs=[inputs], outputs=[final])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[MyMeanIOU()])

    return model


def FitModel(trainingImagesPath: Path, trainingSegmentationsPath: Path, testingImagesPath: Path,
             testingSegmentationsPath: Path, outputPath: Path, epochs: int, batch_size, patience,
             imageSize: typing.Tuple[int, int], dropout_rate, learning_rate):
    print("-----------------------")
    print("Building model...")

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: ' + str(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = CreateModel(dropout_rate, learning_rate, imageSize)
    model.summary()
    print("\tRequired memory: " + str(keras_model_memory_usage_in_bytes(model, batch_size=batch_size)))
    print("\tDone!")

    print("\tImages directory: " + str(trainingImagesPath))
    print("\tSegmentations directory: " + str(trainingSegmentationsPath))
    print("\tModel directory: " + str(outputPath))

    trainingImagePaths = [imagePath for imagePath in sorted(trainingImagesPath.iterdir()) if imagePath.is_file()]
    trainingSegmentationPaths = [segmentationPath for segmentationPath in sorted(trainingSegmentationsPath.iterdir())
                                 if segmentationPath.is_file()]
    testingImagePaths = [imagePath for imagePath in sorted(testingImagesPath.iterdir()) if imagePath.is_file()]
    testingSegmentationPaths = [segmentationPath for segmentationPath in sorted(testingSegmentationsPath.iterdir())
                                if segmentationPath.is_file()]

    print("\tDone.")

    print("\tDone!")

    print(patience)
    earlyStopper = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)
    outputPath.mkdir(parents=True, exist_ok=True)
    print("\tFitting model...", flush=True)
    model.fit(DataGenerator(trainingImagePaths, trainingSegmentationPaths, imageSize, batch_size),
              validation_data=DataGenerator(testingImagePaths, testingSegmentationPaths, imageSize, batch_size),
              verbose=1,
              epochs=epochs,
              callbacks=[PlotCallback(Path(r"C:\Users\jonoj\Documents\ML\AugmentedData\OrganoID_augment_2021_06_24_10_01_40\raw\training\images\337.png"), outputPath / "modelLearningEx"), earlyStopper])
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
