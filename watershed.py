import scipy.ndimage
from PIL import Image
import skimage.measure
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage.segmentation as segmentation
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


def doWatershed(image: np.ndarray, diameter):
    structure = createCircle(diameter)
    distance = ndimage.distance_transform_edt(image)
    localMinima = detect_local_minima(-distance, structure)
    markers, _ = ndimage.label(localMinima)
    labels = segmentation.watershed(-distance, markers, mask=image)
    return labels


def doPostProcess(image: np.ndarray, diameter):
    processed = morphology.remove_small_objects(image, math.pi * ((diameter / 2) ** 2))
    processed = clearBorders(processed, diameter, 1)
    return processed


def clearBorders(image: np.ndarray, border_width, border_thickness):
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
        if touching_borders > prop.major_axis_length*0.75:
            print("Item " + str(prop.label) + ": " + str(touching_borders) + "/" + str(prop.equivalent_diameter))
            toRemove.append(prop.label)

    # mask all label indices that are connected to borders
    label_mask = np.in1d(image, toRemove)
    # create mask for pixels to clear

    mask = label_mask.reshape(image.shape)

    image = image.copy()

    # clear border pixels
    image[mask] = 0

    return image


def plotResults(original, segmented, postProcessed, watershed):
    plt.subplot(1, 4, 1)
    plt.title("Original")
    if original is not None:
        plt.imshow(original)

    plt.subplot(1, 4, 2)
    plt.title("CNN-Segmented")
    if segmented is not None:
        plt.imshow(segmented, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title("Watershed Separation")
    if watershed is not None:
        plt.imshow(watershed, cmap='nipy_spectral')

        props = skimage.measure.regionprops(watershed)
        for prop in props:
            y, x = prop.centroid
            plt.text(x, y, str(prop.label), color="white", ha="center", va="center")

    plt.subplot(1, 4, 4)
    plt.title("Postprocessed")

    if postProcessed is not None:
        plt.imshow(postProcessed, cmap='nipy_spectral')

        props = skimage.measure.regionprops(postProcessed)
        for prop in props:
            y, x = prop.centroid
            plt.text(x, y, str(prop.label), color="white", ha="center", va="center")

    plt.show()


def Watershed(segmented: Image, organoidDiameter=20, plot=True, original: Image = None):
    watershed = doWatershed(segmented, organoidDiameter)
    postProcessed = doPostProcess(watershed, organoidDiameter)
    if plot:
        plotResults(original, segmented, postProcessed, watershed)
    return Image.fromarray(watershed)

#
# testSegmented = Image.open(
#     r"C:\Users\jonoj\Documents\ML\Segmentations\OrganoID_run_2021_06_30_10_16_15\seg_20210211_Aw2PC1.tif").resize(
#     (512, 512))
# Watershed(testSegmented)
