from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))

from backend.ImageManager import LoadImages, SaveImage
from backend.Detector import Detector

image = next(LoadImages(Path(r"dataset\training\images\50.png"))).frames[0]

heat = Detector(Path(r"model\model.tflite")).DetectHeatmap(image)

SaveImage(heat, Path(r"figuresAndStats\overviewFIgure\images\example.png"))