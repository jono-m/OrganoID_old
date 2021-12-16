from pathlib import Path
import sys
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
sys.path.append(str(Path(".").resolve()))

from backend.ImageManager import ShowImage, LoadImages, LabelToRGB, SaveImage
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
labeled = Label(detected, 100, True)

remap = {0: 3,
         1: 6,
         11: 25,
         15: 34,
         25: 26,
         }
remap = {remap[x]: x for x in remap}

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


def circularity(perimeter, area):
    return 4 * np.pi * area / (perimeter ** 2)


areas = [rp.area * squareMicronsPerPixel / 1000 for rp in rps]
circularities = [circularity(rp.perimeter_crofton, rp.area) for rp in rps]
print("ID\tArea\tCircularity")
for rp in rps:
    if rp.label not in remap:
        continue

    (y, x) = rp.centroid
    drawer.text((x, y), str(remap[rp.label]), anchor="mm", fill=(255, 255, 255, 255), font=font)
    print("%d\t%f\t%f" % (
    remap[rp.label], rp.area * squareMicronsPerPixel / 1000, circularity(rp.perimeter_crofton, rp.area)))

SaveImage(detector.DetectHeatmap(image), Path(r"figuresAndStats\singleOrganoidFigure\images\detected.png"))
SaveImage(edges, Path(r"figuresAndStats\singleOrganoidFigure\images\edges.png"))
SaveImage(LabelToRGB(labeled, 0), Path(r"figuresAndStats\singleOrganoidFigure\images\labeled.png"))
SaveImage(np.asarray(merged), Path(r"figuresAndStats\singleOrganoidFigure\images\overlay.png"))


# Based on https://matplotlib.org/stable/gallery/statistics/customized_violin.html

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


plt.subplot(2, 1, 1)
parts = plt.violinplot(areas, showextrema=False, showmeans=False, showmedians=False, vert=False)
for pc in parts['bodies']:
    pc.set_facecolor('#FF1F5B')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, median, quartile3 = np.percentile(areas, [25, 50, 75])
# min area is 100
whisker_min = max(100 * squareMicronsPerPixel / 1000, median - abs(quartile3 - quartile1) * 1.5)
whisker_max = median + abs(quartile3 - quartile1) * 1.5

plt.scatter(median, 1, marker='o', color='white', s=30, zorder=3)
plt.hlines(1, quartile1, quartile3, color='k', linestyle='-', lw=5)
plt.hlines(1, whisker_min, whisker_max, color='k', linestyle='-', lw=1)

for area, rp in zip(areas, rps):
    if area < median + abs(quartile3 - quartile1) * 1.5:
        plt.scatter(area, 1.5, marker='o', color='k', s=10)
        continue
    plt.scatter(area, 1.5, marker='*', color='#FF1F5B', s=10)
    plt.text(area, 1.3, str(remap[rp.label]), horizontalalignment="center", verticalalignment="center",
             color="#FF1F5B",
             size=10)

plt.xlabel(r"Organoid area (x $10^3 \mu m^2$)")

plt.subplot(2, 1, 2)

parts = plt.violinplot(circularities, showextrema=False, showmeans=False, showmedians=False, vert=False)
for pc in parts['bodies']:
    pc.set_facecolor('#AF58BA')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, median, quartile3 = np.percentile(circularities, [25, 50, 75])
whisker_min = median - abs(quartile3 - quartile1) * 1.5
# Max circularity is 1
whisker_max = min(1, median + abs(quartile3 - quartile1) * 1.5)
print(whisker_min)
print(whisker_max)

plt.scatter(median, 1, marker='o', color='white', s=30, zorder=3)
plt.hlines(1, quartile1, quartile3, color='k', linestyle='-', lw=5)
plt.hlines(1, whisker_min, whisker_max, color='k', linestyle='-', lw=1)

for circularity, rp in zip(circularities, rps):
    if circularity > median - abs(quartile3 - quartile1) * 1.5:
        plt.scatter(circularity, 1.5, marker='o', color='k', s=10)
        continue
    plt.scatter(circularity, 1.5, marker='*', color='#AF58BA', s=10)
    plt.text(circularity, 1.3, str(remap[rp.label]), horizontalalignment="center", verticalalignment="center",
             color="#AF58BA",
             size=10)

plt.xlabel("Circularity")

plt.tight_layout()
plt.show()
