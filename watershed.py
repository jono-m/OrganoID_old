from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, watershed_ift, minimum_filter
from pathlib import Path


def bwdistBF(im: np.ndarray):
    distImage = np.zeros(im.shape, np.float)
    for x1 in range(im.shape[0]):
        for y1 in range(im.shape[1]):
            minDistance = None
            for x2 in range(im.shape[0]):
                for y2 in range(im.shape[1]):
                    if im[x2, y2]:
                        d = dist(x1, y1, x2, y2)
                        if minDistance is None or d < minDistance:
                            minDistance = d
            distImage[x1, y1] = minDistance
    return distImage


def bwdistSmarter(im: np.ndarray):
    G = {}
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            minDist = None
            for yP in range(im.shape[1]):
                if im[x, yP]:
                    d = abs(y - yP)
                    if minDist is None or d < minDist:
                        minDist = d
            G[(x, y)] = minDist

    distImage = np.zeros(im.shape, np.float)
    for y in range(im.shape[1]):
        for x in range(im.shape[0]):
            minDist = None
            for xP in range(im.shape[0]):
                g = G[(xP, y)]
                if g is not None:
                    val = (x - xP) ** 2 + G[(xP, y)] ** 2
                    if minDist is None or val < minDist:
                        minDist = val
            distImage[x, y] = minDist ** 0.5
    return distImage


def bwdistSmartest(b: np.ndarray):
    m = b.shape[0]
    n = b.shape[1]
    inf = m + n

    G = np.zeros(b.shape, float)
    for x in range(m):
        if b[x, 0]:
            G[x, 0] = 0
        else:
            G[x, 0] = inf
        for y in range(1, n):
            if b[x, y]:
                G[x, y] = 0
            else:
                G[x, y] = 1 + G[x, y - 1]
        for y in reversed(range(n - 1)):
            if G[x, y + 1] < G[x, y]:
                G[x, y] = 1 + G[x, y + 1]

    distImage = np.zeros(b.shape, np.float)

    for y in range(n):
        def g(i):
            return G[i, y]

        def f(x, i):
            return (x - i) ** 2 + g(i) ** 2

        def Sep(i, u):
            return int((u ** 2 - i ** 2 + g(u) ** 2 - g(i) ** 2) / (2 * (u - i)))

        q = 0
        s = [0] * m
        t = [0] * m
        for u in range(1, m):
            while q >= 0 and f(t[q], s[q]) > f(t[q], u):
                q = q - 1
            if q < 0:
                q = 0
                s[0] = u
            else:
                w = 1 + Sep(s[q], u)
                if w < m:
                    q = q + 1
                    s[q] = u
                    t[q] = w
        for u in reversed(range(m)):
            distImage[u, y] = f(u, s[q])
            if u == t[q]:
                q = q - 1

    return distImage


def bwdistScipy(b: np.ndarray):
    return minimum_filter(-distance_transform_edt(binary_fill_holes(b)), 80)


def dist(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def bwdist(im: Image):
    im = np.array(im)
    bwd = bwdistScipy(im)
    showImage(bwd)


def showImage(im: np.ndarray):
    im = im.astype(float)
    low = np.min(im)
    hi = np.max(im)
    adjust = ((im - low) / (hi - low)) * 256
    adjust = adjust.astype(int)
    Image.fromarray(adjust).show()


filepath = Path(r"C:\Users\jonoj\Documents\ML\Segmentations\OrganoID_run_2021_06_30_10_16_15\seg_20210211_Aw2PC1.tif")

image = Image.open(filepath)
image = image.convert(mode="1")

bwdist(image)
input()
