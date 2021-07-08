import numpy
import scipy.ndimage
from PIL import Image
import skimage.measure
import numpy as np
import math
import scipy.ndimage as ndimage
from scipy.spatial import KDTree
import skimage.segmentation as segmentation
import matplotlib.pyplot as plt
import skimage.morphology as morphology


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
            distance = dist(x, y, center, center)
            if distance <= effectiveDiameter / 2:
                shape[x, y] = True
    return shape


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def mergeClose(image: np.ndarray, separation):
    points = np.argwhere(image)
    newImage = np.zeros(image.shape)

    while True:
        merged = False
        for ia in range(points.shape[0]):
            for ib in range(ia + 1, points.shape[0]):
                pointA = points[ia, :]
                pointB = points[ib, :]
                if dist(pointA[0], pointA[1], pointB[0], pointB[1]) <= separation:
                    merged = True
                    points = np.delete(points, [ia, ib], 0)
                    points = np.append(points, [(pointA + pointB) / 2], 0)
                    break
            if merged:
                break

        if not merged:
            break

    points = numpy.round_(points).astype(int)
    newImage[tuple(points.transpose())] = 1
    return newImage


def doWatershed(image: np.ndarray, separation):
    distance = ndimage.distance_transform_edt(image)
    localMinima = detect_local_minima(-distance, ndimage.generate_binary_structure(2, 2))
    localMinima = mergeClose(localMinima, separation)
    markers, _ = ndimage.label(localMinima)
    labels = segmentation.watershed(-distance, markers, mask=image)
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


def Watershed(segmented: np.ndarray, separation=20):
    return doWatershed(segmented, separation)


def PostProcess(segmented: np.ndarray, areaCutoff, borderCutoff):
    return doPostProcess(segmented, areaCutoff, borderCutoff)
