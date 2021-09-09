import skimage.segmentation as segmentation
import numpy as np
import skimage.feature
import skimage.filters
import scipy.ndimage as ndimage


def Edges(image: np.ndarray, foregroundThreshold, watershedThreshold=None):
    THRESHOLD = 0
    SOBEL = 1

    if watershedThreshold is not None:
        method = THRESHOLD
    else:
        method = SOBEL

    if method == THRESHOLD:
        # The centers of organoids can be found with a higher threshold.
        # Do an opening to remove small debris.
        centers = ndimage.binary_opening(image > watershedThreshold)
        foregroundMask = ndimage.binary_opening(image > foregroundThreshold)
        edges = np.bitwise_and(foregroundMask, np.bitwise_not(centers))
    else:
        # The negated raw output from the CNN is the heightmap for watershed. Organoid borders will be slightly higher
        # than their centers (detected in the initializer image), and so will form the watershed boundary.
        smoothEdges = skimage.filters.gaussian(skimage.filters.sobel(image), 3)

        # The centers of organoids can be found by removing edges.
        edges = skimage.filters.apply_hysteresis_threshold(smoothEdges, 0.003, 0.03)
    return edges


def Label(image: np.ndarray, foregroundThreshold: int, watershedThreshold=None):
    foregroundMask = ndimage.binary_opening(image > foregroundThreshold)
    # The negated raw output from the CNN is the heightmap for watershed. Organoid borders will be slightly higher
    # than their centers (detected in the initializer image), and so will form the watershed boundary.
    smoothForeground = skimage.filters.gaussian(image, 2)
    heightmap = -smoothForeground

    edges = Edges(image, foregroundThreshold, watershedThreshold)

    initializerImage = np.bitwise_and(foregroundMask, np.bitwise_not(edges))

    markers, _ = ndimage.label(initializerImage)
    labels = segmentation.watershed(heightmap, markers, mask=foregroundMask)
    # Some organoids will be lost during the watershed if their initializers were clipped out or too small.
    # Watershed should only really split organoids that are touching, so we want to make sure that the original
    # mask is preserved.
    # First, find all regions that were lost.
    unsplit = np.logical_and(foregroundMask, labels == 0)
    # Label the lost regions
    unsplit_labeled, _ = ndimage.label(unsplit)
    # Make the label numbers for lost regions different from the watershed labels.
    unsplit_labeled[unsplit_labeled > 0] += labels.max() + 1
    # Merge lost regions with the watershed labels.
    labels = labels + unsplit_labeled
    return labels
