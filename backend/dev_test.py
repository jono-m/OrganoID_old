from PIL import Image
from segmentation import SegmentImage, OpenModel
from watershed import Watershed
from pathlib import Path

imagePath = Path(r"C:\Users\jonoj\Documents\ML\aproteinprod.jpg")
modelPath = Path(r"C:\Users\jonoj\Documents\ML\Models\best16\trainedModel")

model = OpenModel(modelPath)
original = Image.open(imagePath)
segmentation = SegmentImage(original, model)
