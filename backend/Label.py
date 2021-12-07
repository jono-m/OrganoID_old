# Label.py -- labels individual organoids in a detection image

import skimage.segmentation as segmentation
import skimage.morphology as morphology
import numpy as np
import skimage.feature
import skimage.filters
import scipy.ndimage as ndimage


def Label(image: np.ndarray, minimumArea: float, removeBorders: bool):
    # Consider organoids to be present at pixels with greater than 50% detection belief.
    foregroundMask = ndimage.binary_opening(image >= 0.5)

    # Watershed algorithm is used to distinguish organoids in contact. The algorithm needs a heightmap and a set of
    # initializer points (basins) for each organoid. The negated detection image is used as the heightmap for watershed
    # (i.e. the organoid centers, which are the strongest predictions, should be at the lowest points in the heightmap).
    smoothForeground = skimage.filters.gaussian(image, 2)
    heightmap = -smoothForeground

    # Basins are found by removing the organoid borders.
    edges = DetectEdges(image)
    centers = np.bitwise_and(foregroundMask, np.bitwise_not(edges))
    basins, _ = ndimage.label(centers)
    labeled = segmentation.watershed(heightmap, basins, mask=foregroundMask)

    # Some small organoids will be lost during the watershed if their edges were relatively too thick
    # to find their centers. Watershed should only really split organoids that are touching, so we want to make sure
    # that organoids in the original mask are preserved.

    # First, find all regions that were lost.
    unsplit = np.logical_and(foregroundMask, labeled == 0)
    # Label the lost regions
    unsplit_labeled, _ = ndimage.label(unsplit)
    # Make the label numbers for lost regions different from the watershed labels.
    unsplit_labeled[unsplit_labeled > 0] += labeled.max() + 1
    # Merge lost regions with the watershed labels.
    labeled = labeled + unsplit_labeled

    # Fill holes in labeled organoids
    labeled = FillHoles(labeled)

    # Remove small organoids
    labeled = morphology.remove_small_objects(labeled, minimumArea)

    if removeBorders:
        labeled = segmentation.clear_border(labeled)

    return labeled


def DetectEdges(image: np.ndarray):
    # Reordered Canny edge detector (Sobel -> Gaussian -> Hysteresis threshold)
    smoothEdges = skimage.filters.gaussian(skimage.filters.sobel(image), 2)
    edges = skimage.filters.apply_hysteresis_threshold(smoothEdges, 0.005, 0.05)

    foregroundMask = ndimage.binary_opening(image >= 0.5)
    edges = np.bitwise_and(edges, foregroundMask)
    return edges


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
