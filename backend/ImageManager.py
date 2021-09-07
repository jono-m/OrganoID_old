import pathlib
from typing import Union, List, Callable
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np
from skimage.color import label2rgb
from skimage.filters import sobel
from backend.Tracker import Tracker


class SmartImage:
    def __init__(self, path: Path, frames: List[np.ndarray]):
        self.path = path
        self.frames = frames

    def DoOperation(self, operation: Callable[[np.ndarray], np.ndarray]):
        return SmartImage(self.path, [operation(frame) for frame in self.frames])


def LoadImages(source: Union[Path, str, List], size=None, recursive=False, mode=None) -> List[SmartImage]:
    if isinstance(source, list):
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
                sort_paths_nicely(source)
            for image in LoadImages(source, size, recursive, mode):
                yield image
        else:
            rawImage = Image.open(source)
            numFrames = getattr(rawImage, "n_frames", 1)
            preparedImages = []
            for frame_number in range(numFrames):
                rawImage.seek(frame_number)

                preparedImage = rawImage

                if mode is not None:
                    if preparedImage.mode[0] == 'I' and mode == "L":
                        preparedImage = preparedImage.convert(mode="I")
                        preparedImage = preparedImage.point(lambda x: x * (1 / 255))

                    if preparedImage.mode != mode:
                        preparedImage = preparedImage.convert(mode=mode)

                if size is not None:
                    preparedImage = preparedImage.resize(size)
                preparedImages.append(np.asarray(preparedImage))

            yield SmartImage(source, preparedImages)
    else:
        raise TypeError("source must be a list, string, or Path-like. Not " + str(type(source)))


def ContrastOp(frame: np.ndarray):
    return 255 * ((frame - frame.min()) / (frame.max() - frame.min()))


def ShowImage(frame: np.ndarray):
    Image.fromarray(frame).show()


def SaveGIF(images: List[np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(images[0]).save(path, save_all=True, append_images=[Image.fromarray(im) for im in images[1:]],
                                    loop=0)


def SaveTIFFStack(images: List[np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(images[0]).save(path, save_all=True, append_images=[Image.fromarray(im) for im in images[1:]],
                                    compression=None)


def SaveImage(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def LabelToRGB(image: np.ndarray):
    labeled = (label2rgb(image, bg_label=0) * 255).astype(np.uint8)

    return labeled


def LabelTracks(tracks: List[Tracker.OrganoidTrack], labelColor, outlineAlpha, fillAlpha, presentColor, missingColor,
                baseImages):
    images = []

    font = ImageFont.truetype("arial.ttf", 16)

    for frame, baseImage in enumerate(baseImages):
        print("Labeling " + str(frame))
        tracksAtFrame = [track for track in tracks if track.DataAtFrame(frame) is not None]

        pilImage = Image.new(mode="RGBA", size=baseImage.shape, color=(0, 0, 0, 0))
        drawer = ImageDraw.Draw(pilImage)

        for track in tracksAtFrame:
            data = track.DataAtFrame(frame)
            fillCoords = list(zip(list(data.pixels[:, 1]), list(data.pixels[:, 0])))

            if data.detection:
                color = presentColor
            else:
                color = missingColor

            drawer.point(fillCoords, color + (fillAlpha,))
            borderCoords = ComputeOutline(data.image)
            globalCoords = borderCoords + data.bbox[:2]
            xs = list(globalCoords[:, 1])
            ys = list(globalCoords[:, 0])
            outlineCoords = list(zip(xs, ys))
            drawer.point(outlineCoords, color + (outlineAlpha,))

            y, x = list(data.centroid)
            drawer.text((x, y), str(track.id), anchor="ms", fill=labelColor, font=font)

        baseImage = Image.fromarray(baseImage).convert(mode="RGBA")
        pilImage = Image.alpha_composite(baseImage, pilImage)
        images.append(np.asarray(pilImage))
    return images


def ComputeOutline(image: np.ndarray):
    edge = sobel(image, mode="constant")
    coords = np.argwhere(np.greater(edge, 0.5))
    return coords


import re


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def path_alphanum_key(p: Path):
    return alphanum_key(p.name)


def sort_paths_nicely(paths: List[pathlib.Path]):
    paths.sort(key=path_alphanum_key)
