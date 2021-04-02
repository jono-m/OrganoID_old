import training
from pathlib import Path

training.FitModel(trainingImagesPath=Path(r'C:\Users\jonoj\Documents\ML\OrganoID_augment_2021_04_01_19_09_21\images'),
                  trainingSegmentationsPath=Path(r'C:\Users\jonoj\Documents\ML\OrganoID_augment_2021_04_01_19_09_21\segmentations'),
                  outputPath=Path(r'C:\Users\jonoj\Documents\ML\Model'),
                  epochs=10,
                  test_size=0.2,
                  batch_size=8,
                  patience=5,
                  imageSize=(512, 512),
                  numImages=2000,
                  dropout_rate=0.1,
                  learning_rate=0.001)
