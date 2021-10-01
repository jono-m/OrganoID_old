import skimage.measure
import skimage.morphology as morphology
import scipy.ndimage as ndimage
import numpy as np


def PostProcess(image: np.ndarray, minimumArea: float = None, borderDiameterCutoff: float = None,
                fillOrganoidHoles=True):
    # Fill holes of contiguous regions.
    if fillOrganoidHoles:
        image = FillHoles(image)

    # Remove debris.
    if minimumArea:
        image = morphology.remove_small_objects(image, minimumArea)

    # Remove organoids that are significantly clipped by the image borders.
    if borderDiameterCutoff is not None:
        image = clearBorders(image, borderDiameterCutoff)

    return image


def clearBorders(image: np.ndarray, borderDiameterCutoff: float):
    # Find organoids in the image.
    toRemove = []
    props = skimage.measure.regionprops(image)
    for prop in props:
        # The organoid will be removed if more than borderDiameterCutoff * effective_diameter pixels are touching
        # a border.
        cutoff = prop.major_axis_length * borderDiameterCutoff

        touchingBorders = [np.count_nonzero(prop.coords[:, dim] == 0) for dim in range(image.ndim)] + \
                          [np.count_nonzero(prop.coords[:, dim] == (image.shape[dim] - 1)) for dim in range(image.ndim)]

        if any([touchingBorder > cutoff for touchingBorder in touchingBorders]):
            toRemove.append(prop.label)

    image[np.isin(image, toRemove)] = 0

    return image


def FillHoles(image: np.ndarray):
    filledImage = np.zeros_like(image)
    labels = np.unique(image)
    for label in labels:
        if label == 0:
            continue
        subImage = np.where(image == label, True, False)
        filled = ndimage.binary_fill_holes(subImage)
        filledImage[filled] = label

    return filledImage
