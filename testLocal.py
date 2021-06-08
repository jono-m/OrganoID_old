import training
from pathlib import Path

imagesPath = Path("C:/Users/jonoj/Documents/ML/2021_OriginalX")
segmentationsPath = Path("C:/Users/jonoj/Documents/ML/202105_OriginalY")
training.FitModel(imagesPath, segmentationsPath, Path("C:/Models"), epochs=5, test_size=0, patience=5,
                  batch_size=1, imageSize=(512, 512), dropout_rate=0, learning_rate=0.001)
