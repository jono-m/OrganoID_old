from typing import Union, List
from PIL import Image
from pathlib import Path
import numpy as np
import itertools


class SmartImage:
    def __init__(self, path: Path, image: np.ndarray, number):
        self.path = path
        self.image = image
        self.number = number


def LoadImages(source: Union[Path, str, List], size=None, recursive=False, mode='L'):
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

                if preparedImage.mode == 'I' or preparedImage.mode == 'I;16':
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


def SaveImage(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
