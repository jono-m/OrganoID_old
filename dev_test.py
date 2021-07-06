from PIL import Image
from segmentation import SegmentImage, OpenModel
from watershed import Watershed
from pathlib import Path

imagePath = Path(r"C:\Users\jonoj\Documents\ML\RawData\images\334.png")
modelPath = Path(r"C:\Users\jonoj\Documents\ML\Models\OrganoID_train_2021_06_29_23_58_39\trainedModel")

model = OpenModel(modelPath)
original = Image.open(imagePath)
segmentation = SegmentImage(original, model)
watershed = Watershed(segmentation, original=original)
