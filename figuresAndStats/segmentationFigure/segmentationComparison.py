import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(".").resolve()))

from backend.ImageManager import ShowImage, SaveImage, LoadImages
from backend.Detector import Detector
from PIL import Image

exampleImages = ["ACC3", "C3", "Lung5", "PDAC9"]
images = [
    list(LoadImages(Path(r"dataset\testing\images\%s*" % name), mode="L"))[0].Resize([512, 512]).frames[0] for name
    in exampleImages]
groundTruthImages = [
    list(LoadImages(Path(r"dataset\testing\segmentations\%s*" % name), mode="1"))[0].Resize([512, 512]).frames[0] for name
    in exampleImages]

outPath = Path(r"figuresAndStats\segmentationFigure\images")

detector = Detector(Path(r"model\model.tflite"))
for name, image, groundTruth in zip(exampleImages, images, groundTruthImages):
    detected = detector.Detect(image)
    groundTruthColored = np.zeros([512, 512, 3], dtype=np.uint8)
    detectedColored = np.zeros([512, 512, 3], dtype=np.uint8)
    merged = np.zeros([512, 512, 3], dtype=np.uint8)

    groundTruthColored[np.where(groundTruth)] = [174, 205, 236]
    merged[np.where(groundTruth)] = [174, 205, 236]

    detectedColored[np.where(detected > 0.5)] = [243, 213, 123]
    merged[np.where(detected > 0.5)] = [243, 213, 123]

    merged[np.where(np.bitwise_and(detected >= 0.5, groundTruth))] = [112, 200, 120]

    SaveImage(merged, outPath / (name + "_merged.png"))
    SaveImage(detectedColored, outPath / (name + "_detected.png"))
    SaveImage(groundTruthColored, outPath / (name + "_gt.png"))