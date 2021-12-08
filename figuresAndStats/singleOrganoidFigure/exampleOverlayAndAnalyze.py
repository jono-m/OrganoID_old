from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(".").resolve()))

from backend.ImageManager import LoadImages, ShowImage, LabelToRGB
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from skimage.measure import regionprops

imageFile = Path(r"figuresAndStats\exampleOverlay")
labeledFile = Path(r"dataset\demo\labeled\*labeled*")

image = list(LoadImages(imageFile, size=(512, 512), mode="L"))[0].frames[0]
labeled = list(LoadImages(labeledFile))[0].frames[26]

rps = regionprops(labeled)

focusOrganoidsLabels = [3, 6, 7, 8, 9, 11, 12, 15, 17, 12]
idMapToTrackingFigure = [0, 1, 2, 3, 5, 6, 9, 18, 33]

transparency = np.where(np.not_equal(labeled, 0), 128, 0)
overlay = LabelToRGB(labeled, 0)
overlay = np.append(overlay, transparency[:, :, None], axis=2).astype(np.uint8)
overlay = Image.fromarray(overlay)
composite = Image.alpha_composite(Image.fromarray(image).convert(mode="RGBA"), overlay).convert(mode="RGB")

font = ImageFont.truetype("arial.ttf", 30)

drawer = ImageDraw.Draw(composite)
focusOrganoids = [rp for rp in rps if rp.label in focusOrganoidsLabels]
focusOrganoids.sort(key=lambda x: x.area)
for focusOrganoid in focusOrganoids:
    index = focusOrganoidsLabels.index(focusOrganoid.label)
    idToDraw = idMapToTrackingFigure[index]
    (y, x) = focusOrganoid.centroid
    drawer.text((x, y), str(idToDraw), anchor="mm", fill=(255, 255, 255, 255), font=font)

# ShowImage(np.asarray(composite))

conversion = 6.8644  # microns^2/pixel

plt.boxplot([prop.area * conversion for prop in rps], vert=False)
plt.xlabel(r"Organoid Area ($\mu ^2$)")
selectedOrganoids = []
plt.scatter([focusOrganoid.area * conversion for focusOrganoid in focusOrganoids], [1 for i in range(len(focusOrganoids))])
for i, focusOrganoid in enumerate(focusOrganoids):
    index = focusOrganoidsLabels.index(focusOrganoid.label)
    idToDraw = idMapToTrackingFigure[index]
    plt.text(focusOrganoid.area * conversion, 1 + i * 0.1, str(idToDraw), verticalalignment='center',
             horizontalalignment='left',
             color="black")
    print("%d: %d" % (idToDraw, focusOrganoid.area * conversion))
plt.show()
