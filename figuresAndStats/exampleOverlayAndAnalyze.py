from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))

from backend.ImageManager import LoadImages, ShowImage, LabelToRGB
from PIL import Image
import numpy as np
from skimage.measure import regionprops

imageFile = Path(r"dataset\demo\20210127_organoidplate003_XY36_Z3_C2.tif")
labeledFile = Path(r"dataset\demo\labeled\*_rgb*")

image = list(LoadImages(imageFile, size=(512, 512), mode="L"))[0].frames[29] * 2
labeled = list(LoadImages(labeledFile))[0].frames[29]

rp = regionprops(labeled)

focusOrganoids = [1, 2, 3]

overlay =

overlay = Image.fromarray(overlay).convert(mode="RGBA")
overlay.putalpha(128)
underlay = Image.alpha_composite(Image.fromarray(image).convert(mode="RGBA"), overlay).convert(mode="RGB")
ShowImage(np.asarray(underlay))
