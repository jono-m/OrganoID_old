import numpy
from time import time
import skimage.measure
import skimage.segmentation as segmentation
import skimage.morphology as morphology
import numpy as np
import math
import scipy.ndimage as ndimage


def detect_local_minima(arr, structure):
    minFilter = ndimage.minimum_filter(arr, footprint=structure) == arr
    background = arr == 0
    foreground = minFilter & ~background
    return foreground


def createCircle(diameter):
    effectiveDiameter = math.ceil(diameter)
    effectiveDiameter = effectiveDiameter + ((effectiveDiameter + 1) % 2)
    center = math.floor(effectiveDiameter / 2)
    shape = np.zeros((effectiveDiameter, effectiveDiameter), dtype=bool)
    for x in range(effectiveDiameter):
        for y in range(effectiveDiameter):
            distance = dist((x, y), (center, center))
            if distance <= effectiveDiameter / 2:
                shape[x, y] = True
    return shape


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def findCenter(raw: np.ndarray, centerThreshold: float):
    return raw > centerThreshold


def doWatershed(image: np.ndarray, raw: np.ndarray, centerThreshold: float, separation):
    distance = ndimage.distance_transform_edt(image)
    localMinima = findCenter(raw, centerThreshold)
    if int(separation) > 0:
        localMinima = ndimage.binary_opening(localMinima, iterations=int(separation))
    markers, _ = ndimage.label(localMinima)
    labels = segmentation.watershed(-distance, markers, mask=image)
    unsplit = np.logical_and(image, labels == 0)
    unsplit_labeled, _ = ndimage.label(unsplit)
    unsplit_labeled[unsplit_labeled > 0] += labels.max()
    labels = labels + unsplit_labeled
    return distance, markers, labels


def doPostProcess(image: np.ndarray, area, borderCutoff):
    processed = morphology.remove_small_objects(image, area)
    if borderCutoff > 0:
        processed = clearBorders(processed, borderCutoff, 1)
    return processed


def clearBorders(image: np.ndarray, borderCutoff, border_thickness):
    # create borders with buffer_size
    borders = np.zeros_like(image, dtype=bool)
    slstart = slice(border_thickness)
    slend = slice(-border_thickness, None)
    slices = [slice(s) for s in image.shape]
    for d in range(image.ndim):
        slicedim = list(slices)
        slicedim[d] = slstart
        borders[tuple(slicedim)] = True
        slicedim[d] = slend
        borders[tuple(slicedim)] = True

    toRemove = []
    props = skimage.measure.regionprops(image)

    for prop in props:
        touching_borders = np.count_nonzero(borders[prop.coords[:, 0], prop.coords[:, 1]])
        if touching_borders > prop.major_axis_length * (1 - borderCutoff):
            toRemove.append(prop.label)

    # mask all label indices that are connected to borders
    label_mask = np.in1d(image, toRemove)
    # create mask for pixels to clear

    mask = label_mask.reshape(image.shape)

    image = image.copy()

    # clear border pixels
    image[mask] = 0

    return image


def Watershed(segmented: np.ndarray, raw, centerThreshold, separation):
    return doWatershed(segmented, raw, centerThreshold, separation)


def PostProcess(segmented: np.ndarray, areaCutoff, borderCutoff):
    return doPostProcess(segmented, areaCutoff, borderCutoff)
