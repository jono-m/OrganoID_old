import skimage.segmentation as segmentation
import numpy as np
import scipy.ndimage as ndimage


def Label(image: np.ndarray, foregroundThreshold: int, watershedThreshold: int):
    foregroundImage = np.where(image > foregroundThreshold, image, 0)
    foregroundMask = ndimage.binary_opening(image > foregroundThreshold)

    if watershedThreshold > foregroundThreshold:
        # The centers of organoids can be found with a higher threshold.
        # Do an opening to remove small debris.
        initializerImage = ndimage.binary_opening(foregroundImage > watershedThreshold)
        markers, _ = ndimage.label(initializerImage)
        # The negated raw output from the CNN is the heightmap for watershed. Organoid borders will be slightly higher than
        # their centers (detected in the initializer image), and so will form the watershed boundary.
        heightmap = -foregroundImage
        labels = segmentation.watershed(heightmap, markers, mask=foregroundMask)
        # Some organoids will be lost during the watershed if their initializers were clipped out or too small. Watershed
        # should only really split organoids that are touching, so we want to make sure that the original mask is preserved.
        # First, find all regions that were lost.
        unsplit = np.logical_and(foregroundMask, labels == 0)
        # Label the lost regions
        unsplit_labeled, _ = ndimage.label(unsplit)
        # Make the label numbers for lost regions different from the watershed labels.
        unsplit_labeled[unsplit_labeled > 0] += labels.max() + 1
        # Merge lost regions with the watershed labels.
        labels = labels + unsplit_labeled
    else:
        labels, _ = ndimage.label(foregroundMask)

    return labels
