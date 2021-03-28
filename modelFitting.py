from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.metrics import mean_iou
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
import typing


def MeanIoU(y_true, y_pred):
    return mean_iou(y_true, y_pred, 2)


def FitModel(trainingImagesPath: Path, trainingSegmentationsPath: Path, outputPath: Path, epochs: int,
             test_size=0.5, batch_size=1, patience=5, imageSize: typing.Tuple[int, int] = (640, 640)):
    print("-----------------------")
    print("Building model...")
    print("\tImages directory: " + str(trainingImagesPath))
    print("\tSegmentations directory: " + str(trainingSegmentationsPath))
    print("\tModel directory: " + str(outputPath))

    print("\tLoading images...")
    images = [Image.open(imagePath) for imagePath in sorted(trainingImagesPath.iterdir()) if imagePath.is_file()]

    segmentations = [Image.open(segmentationPath) for segmentationPath in sorted(trainingSegmentationsPath.iterdir())
                     if segmentationPath.is_file()]

    print("\tDone.")

    for imageIndex, image in enumerate(images):
        print("\tConverting image " + str(imageIndex + 1) + "/" + str(len(images)))
        if image.mode == 'I':
            image = image.point(lambda x: x * (1 / 255))
        images[imageIndex] = np.array(image.resize(imageSize).convert(mode="L"))
    images = np.expand_dims(np.moveaxis(np.stack(images, axis=-1), -1, 0), -1)

    print("Images Dim: " + str(images.shape))
    print("\tDone!")

    for segmentationindex, segmentation in enumerate(segmentations):
        print("\tConverting segmentation " + str(segmentationindex + 1) + "/" + str(len(segmentations)))
        segmentations[segmentationindex] = np.array(segmentation.resize(imageSize).convert(mode="1"))

    segmentations = np.expand_dims(np.moveaxis(np.stack(segmentations, axis=-1), -1, 0), -1)
    print("Seg Dim: " + str(images.shape))

    print("\tDone!")

    print("\tSplitting training and testing datasets (" + str(test_size * 100) + "% for testing)...")
    trainingImages, testingImages, trainingSegmentations, testingSegmentations = train_test_split(images,
                                                                                                  segmentations,
                                                                                                  test_size=test_size)

    print("\tDone!")

    print("\tBuilding model pipeline...")
    inputs = Input((imageSize[0], imageSize[1], 1))

    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print("\tDone!")

    earlystopper = EarlyStopping(patience=patience, verbose=1)
    print("\tFitting model...")
    model.fit(x=trainingImages, y=trainingSegmentations,
              validation_data=(testingImages, testingSegmentations),
              batch_size=batch_size,
              verbose=1,
              epochs=epochs,
              callbacks=[earlystopper])
    print("\tDone!")

    print("\tSaving model...")

    modelJobSavePath = outputPath / "trainedModel.h5"

    model.save(modelJobSavePath)

    results = modelJobSavePath
    print("Model saved to " + str(results))

    return results
