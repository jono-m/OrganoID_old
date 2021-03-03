from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.metrics import mean_iou
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path


def FitModel(jobID: str, trainingImagesPath: Path, trainingSegmentationsPath: Path, modelSavePath: Path, epochs: int,
             test_size=0.5, batch_size=32, patience=5):
    print("-----------------------")
    print("Building model...")
    print("\tImages directory: " + str(trainingImagesPath))
    print("\tSegmentations directory: " + str(trainingSegmentationsPath))
    print("\tModel directory: " + str(modelSavePath))

    modelJobSavePath = modelSavePath / ("OrganoID_model_" + jobID)

    print("\tLoading images...")
    images = [Image.open(imagePath) for imagePath in sorted(trainingImagesPath.iterdir()) if imagePath.is_file()]

    segmentations = [Image.open(segmentationPath) for segmentationPath in sorted(trainingSegmentationsPath.iterdir())
                     if segmentationPath.is_file()]

    print("\tDone.")
    imageSize = images[0].size

    for imageIndex, image in enumerate(images):
        print("\tConverting image " + str(imageIndex + 1) + "/" + str(len(images)))
        if image.mode == 'I':
            image = image.point(lambda x: x * (1 / 255))
        images[imageIndex] = np.array(image.resize(imageSize).convert(mode="RGB"))
    images = np.moveaxis(np.stack(images, axis=-1), -1, 0)

    print("\tDone!")

    for segmentationindex, segmentation in enumerate(segmentations):
        print("\tConverting segmentation " + str(segmentationindex + 1) + "/" + str(len(segmentations)))
        segmentations[segmentationindex] = np.array(segmentation.resize(imageSize).convert(mode="1"))

    segmentations = np.moveaxis(np.stack(segmentations, axis=-1), -1, 0)

    print("\tDone!")

    print("\tSplitting training and testing datasets (" + str(test_size * 100) + "% for testing)...")
    trainingImages, testingImages, trainingSegmentations, testingSegmentations = train_test_split(images,
                                                                                                  segmentations,
                                                                                                  test_size=test_size)

    print("\tDone!")

    print("\tBuilding model pipeline...")
    inputs = Input((imageSize[0], imageSize[1], 3))
    s = Lambda(lambda x: x / 255)(inputs)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou, 'accuracy'])
    model.summary()
    print("\tDone!")

    earlystopper = EarlyStopping(patience=patience, verbose=1)
    print("\tRunning model...")
    model.fit(x=trainingImages, y=trainingSegmentations,
              validation_data=(testingImages, testingSegmentations),
              batch_size=batch_size,
              verbose=1,
              epochs=epochs,
              callbacks=[earlystopper])
    print("\tDone!")

    print("\tSaving model...")
    model.save(modelJobSavePath)

    results = modelJobSavePath
    print("Model saved to " + str(results))

    return results
