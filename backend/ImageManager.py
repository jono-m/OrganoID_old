from typing import Union, List, Tuple
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.filters import sobel


class SmartImage:
    def __init__(self, path: Path, image: np.ndarray, number):
        self.path = path
        self.image = image
        self.number = number


def LoadImages(source: Union[Path, str, List], size=None, recursive=False, mode=None):
    if isinstance(source, List):
        for i in source:
            for image in LoadImages(i, size, recursive, mode):
                yield image
    elif isinstance(source, str):
        for image in LoadImages(Path(source), size, recursive, mode):
            yield image
    elif isinstance(source, Path):
        if source.is_dir():
            if recursive:
                source = [path for path in source.iterdir()]
            else:
                source = [path for path in source.iterdir() if path.is_file()]
            for image in LoadImages(source, size, recursive, mode):
                yield image
        else:
            rawImage = Image.open(source)
            preparedImages = []
            for frame_number in range(rawImage.n_frames):
                rawImage.seek(frame_number)

                preparedImage = rawImage
                if size is not None:
                    preparedImage = preparedImage.resize(size)

                if mode is not None:
                    if preparedImage.mode == 'I' or preparedImage.mode == 'I;16' and mode == "L":
                        preparedImage = preparedImage.point(lambda x: x * (1 / 255))

                    if preparedImage.mode != mode:
                        preparedImage = preparedImage.convert(mode=mode)
                preparedImages.append((frame_number, np.asarray(preparedImage)))
            for (number, image) in preparedImages:
                yield SmartImage(source, image, number)
    else:
        raise TypeError("source must be of type List, string, or Path-like. Not " + str(type(source)))


def Contrast(image: np.ndarray):
    return 255 * ((image - image.min()) / (image.max() - image.min()))


def ShowImage(image: np.ndarray):
    Image.fromarray(image).show()


def SaveImages(images: List[np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(images[0]).save(path, save_all=True, append_images=[Image.fromarray(im) for im in images[1:]],
                                    loop=0)


def SaveImage(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def LabelToRGB(image: np.ndarray, overlay=False):
    labeled = (label2rgb(image, bg_label=0) * 255).astype(np.uint8)

    return labeled


def LabelToColor(image: np.ndarray, color):
    colored = np.zeros(image.shape + [3], dtype=np.uint8)
    colored[image > 0] = color
    return colored


def EmphasizeLabeled(labels: np.ndarray, labelColor=None, outlineColor=None):
    pilImage = Image.fromarray(np.zeros_like(labels)).convert(mode="RGB")
    drawer = ImageDraw.Draw(pilImage)
    font = ImageFont.truetype("arial.ttf", 16)
    props = regionprops(labels)
    for prop in props:
        borderCoords = ComputeOutline(prop.image)
        globalCoords = borderCoords + prop.bbox[:2]
        xs = list(globalCoords[:, 1])
        ys = list(globalCoords[:, 0])
        outlineCoords = list(zip(xs, ys))
        drawer.point(outlineCoords, outlineColor)

        y, x = list(prop.centroid)
        drawer.text((x, y), str(prop.label), anchor="ms", fill=labelColor, font=font)

    return np.asarray(pilImage)


def ComputeOutline(image: np.ndarray):
    edge = sobel(image, mode="constant")
    coords = np.argwhere(edge > 0.5)
    return coords
