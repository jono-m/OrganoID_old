# ImageManager.py -- provides a framework for loading and manipulating image frames irrespective of file type,
# and whether the loaded path is a regex, directory, image, or image stack. Also some utility functions for image
# manipulation and display

import pathlib
from typing import Union, List, Callable
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np
import sys
import re
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.measure import regionprops
from backend.Tracker import Tracker
from util import Printer


# A SmartImage maintains knowledge about the path that an image was loaded from (and groups stacks together)
class SmartImage:
    def __init__(self, path: Path, frames: List[np.ndarray]):
        self.path = path
        self.frames = frames

    def DoOperation(self, operation: Callable[[np.ndarray], np.ndarray], verbose=False):
        images = []
        for i, frame in enumerate(self.frames):
            if verbose:
                Printer.printRep("%d/%d" % (i+1, len(self.frames)))
            images.append(operation(frame))
        if verbose:
            Printer.printRep()
        return SmartImage(self.path, images)


# Lazy load of images from a string or Path-like object of a directory, file, or stack.
# size: A tuple (width, height) -- resize loaded images to this size
# mode: the image mode to convert all loaded images into (from PIL library, e.g. "L", "1", "RGB")
def LoadImages(source: Union[Path, str, List], size=None, mode=None) -> List[SmartImage]:
    if isinstance(source, list):
        # Iterate through path lists
        for i in source:
            for image in LoadImages(i, size, mode):
                yield image
    elif isinstance(source, str):
        # Convert strings to Path-like
        for image in LoadImages(Path(source), size, mode):
            yield image
    elif isinstance(source, Path):
        if source.is_dir():
            # Load directories
            source = [path for path in source.iterdir() if path.is_file()]
            # Sort alphabetically and respect numbering (important for image tracking)
            sort_paths_nicely(source)
            for image in LoadImages(source, size, mode):
                yield image
        else:
            if not source.is_file():
                # Handle regular expression paths
                regex = source.name
                directory = source.parent
                source = [path for path in directory.glob(regex) if path.is_file()]
                sort_paths_nicely(source)
                for image in LoadImages(source, size, mode):
                    yield image
                return

            try:
                rawImage = Image.open(source)
            except Exception as e:
                print("Could not load. Error " + str(e), file=sys.stderr)
                return
            numFrames = getattr(rawImage, "n_frames", 1)
            preparedImages = []
            # Load image, which possibly has multiple frames (e.g. TIFF stack)
            for frame_number in range(numFrames):
                rawImage.seek(frame_number)

                preparedImage = rawImage

                if mode is not None:
                    if preparedImage.mode[0] == 'I' and mode == "L":
                        # Intensity images (i.e. 16-bit or 32-bit floating-point) should be divided by 255 to convert to
                        # 8-bit.
                        preparedImage = preparedImage.convert(mode="I")
                        preparedImage = preparedImage.point(lambda x: x * (1 / 255))

                    if preparedImage.mode != mode:
                        # Convert to expected mode
                        preparedImage = preparedImage.convert(mode=mode)

                if size is not None:
                    # Resize image
                    preparedImage = preparedImage.resize(size)
                preparedImages.append(np.asarray(preparedImage))

            yield SmartImage(source, preparedImages)
    else:
        raise TypeError("source must be a list, string, or Path-like. Not " + str(type(source)))


# Displays a numpy array as an image
def ShowImage(frame: np.ndarray):
    Image.fromarray(frame).show()


# Saves a list of images as a GIF.
def SaveGIF(images: List[np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(images[0]).save(path, save_all=True, append_images=[Image.fromarray(im) for im in images[1:]],
                                    loop=0)


# Saves a list of images as a TIFF stack)
def SaveTIFFStack(images: List[np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(images[0]).save(path, save_all=True, append_images=[Image.fromarray(im) for im in images[1:]],
                                    compression=None)


# Save a single image
def SaveImage(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


# Convert a numberically labeled image to a randomly-colored RGB image (with optional drawing of label number on
# each island.
def LabelToRGB(image: np.ndarray, textSize):
    # Scikit-image RGB is 0-1. Convert to 8-bit RGB.
    labeled = (label2rgb(image, bg_label=0) * 255).astype(np.uint8)

    if textSize:
        font = ImageFont.truetype("arial.ttf", textSize)
        regionProperties = regionprops(image)

        withTextImage = Image.fromarray(labeled)
        drawer = ImageDraw.Draw(withTextImage)
        for regionProperty in regionProperties:
            (y, x) = regionProperty.centroid
            drawer.text((x, y), str(regionProperty.label), anchor="ms", fill=(255, 255, 255, 255), font=font)
        labeled = np.asarray(withTextImage)
    return labeled


# Overlay a set of organoid tracks on a list of base images.
def LabelTracks(tracks: List[Tracker.OrganoidTrack], labelColor, outlineAlpha, fillAlpha, presentColor, missingColor,
                baseImages):
    images = []

    font = ImageFont.truetype("arial.ttf", 26)

    for frame, baseImage in enumerate(baseImages):
        # Find the tracks that are at the given frame number
        tracksAtFrame = [track for track in tracks if track.DataAtFrame(frame) is not None]

        pilImage = Image.new(mode="RGBA", size=baseImage.shape, color=(0, 0, 0, 0))
        drawer = ImageDraw.Draw(pilImage)

        # Draw each present track on the frame
        for track in tracksAtFrame:
            data = track.DataAtFrame(frame)
            fillCoords = list(zip(list(data.pixels[:, 1]), list(data.pixels[:, 0])))

            if data.wasDetected:
                color = presentColor
            else:
                continue

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
        pilImage = Image.alpha_composite(baseImage, pilImage).convert(mode="RGB")
        images.append(np.asarray(pilImage))
    return images


def ComputeOutline(image: np.ndarray):
    # Finds the outline of an image.
    edge = sobel(image, mode="constant")
    coords = np.argwhere(np.greater(edge, 0.5))
    return coords


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
