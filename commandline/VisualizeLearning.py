from pathlib import Path
from backend.ImageManager import LoadImages
from backend.Segmenter import Segmenter
import matplotlib.pyplot as plt
import numpy as np
import imageio

import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


raw = LoadImages(r"C:\Users\jonoj\Documents\ML\RawData\images\xy1.png", size=(512, 512))[0]

modelDirectory = Path(r"C:\Users\jonoj\Repositories\OrganoID\assets\newBest\modelEpochs")

modelPaths = sorted(list(modelDirectory.iterdir()), key=lambda x: natural_keys(x.stem))

images = []

for path in modelPaths:
    segmenter = Segmenter(path)
    segmented = segmenter.Segment(raw)

    tinted = (plt.get_cmap("hot")(segmented)[:, :, :3] * 255).astype(np.uint8)

    images.append(tinted)
    print(path)

imageio.mimsave('./test.gif', images)
