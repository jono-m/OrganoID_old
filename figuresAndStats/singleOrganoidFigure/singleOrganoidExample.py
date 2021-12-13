from pathlib import Path
import sys
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
sys.path.append(str(Path(".").resolve()))

from backend.ImageManager import LoadImages, LabelToRGB, SaveImage
from backend.Detector import Detector
from backend.Label import Label, DetectEdges

from PIL import Image, ImageFont, ImageDraw
import numpy as np
from skimage.measure import regionprops

imageFile = Path(r"dataset\demo")
image = list(LoadImages(imageFile, size=(512, 512), mode="L"))[0].frames[26]

detector = Detector(Path(r"model\model.tflite"))
detected = detector.Detect(image)
edges = DetectEdges(detected)
labeled = Label(detected, 100, False)

remap = {6: 1, 8: 3, 11: 6, 9: 5, 7: 2, 3: 0, 15: 18, 12: 9}

alphaMap = np.where(labeled == 0, 0, 128)
overlay = LabelToRGB(labeled, 0)
overlay = np.append(overlay, alphaMap[:, :, None], axis=2).astype(np.uint8)
overlay = Image.fromarray(overlay)
underlay = Image.fromarray(image * 2).convert(mode="RGBA")
merged = Image.alpha_composite(underlay, overlay)

SaveImage(np.asarray(merged), Path(r"figuresAndStats\singleOrganoidFigure\images\overlayNoText.png"))

font = ImageFont.truetype("arial.ttf", 30)
drawer = ImageDraw.Draw(merged)
rps = regionprops(labeled)

squareMicronsPerPixel = 6.8644

areas = [rp.area * squareMicronsPerPixel / 1000 for rp in rps]
for rp in rps:
    if rp.label not in remap:
        continue

    (y, x) = rp.centroid
    drawer.text((x, y), str(remap[rp.label]), anchor="mm", fill=(255, 255, 255, 255), font=font)

plt.violinplot(areas, showextrema=False, vert=False)
plt.plot(areas, [1 for _ in areas], 'o', color="black")
plt.boxplot(areas, vert=False)
plt.xlabel(r"Organoid area (x $10^3 \mu m^2$)")
plt.show()

SaveImage(detector.DetectHeatmap(image), Path(r"figuresAndStats\singleOrganoidFigure\images\detected.png"))
SaveImage(edges, Path(r"figuresAndStats\singleOrganoidFigure\images\edges.png"))
SaveImage(LabelToRGB(labeled, 0), Path(r"figuresAndStats\singleOrganoidFigure\images\labeled.png"))
SaveImage(np.asarray(merged), Path(r"figuresAndStats\singleOrganoidFigure\images\overlay.png"))
