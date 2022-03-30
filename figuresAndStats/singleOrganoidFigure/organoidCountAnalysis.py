import sys

from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from backend.Detector import Detector
from backend.ImageManager import LoadImages
from backend.Label import Label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndimage


modelPath = Path(r"model\model.tflite")
imagesPath = Path(r"dataset\testing\images")
segmentationsPath = Path(r"dataset\testing\segmentations")

detector = Detector(modelPath)

images = LoadImages(imagesPath, mode="L")
segmentations = LoadImages(segmentationsPath, mode="1")

organoID_counts = []
manual_counts = []

file = open(r"figuresAndStats\singleOrganoidFigure\data\counts.csv", "w+")
file.write("Filename, Manual count, OrganoID count\n")
for (image, segmentation) in zip(images, segmentations):
    print("Analyzing image %s" % image.path.name)
    detected = detector.Detect(image.frames[0])
    organoID_labeled = Label(detected, 200, False)

    manual_labeled, _ = ndimage.label(segmentation.frames[0])
    manual_labeled = remove_small_objects(manual_labeled, 200)

    num_organoID = len(regionprops(organoID_labeled))
    num_manual = len(regionprops(manual_labeled))

    organoID_counts.append(num_organoID)
    manual_counts.append(num_manual)
    file.write("%s, %d, %d\n" % (image.path.name, num_manual, num_organoID))
file.close()