import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def SmoothCurve(x, y, resolution):
    # 300 represents number of points to make between T.min and T.max
    xSmooth = np.linspace(0, max(x) - min(x), (max(x) - min(x)) * resolution + 1) + min(x)

    spline = make_interp_spline(x, y, k=3)  # type: BSpline
    ySmooth = spline(xSmooth)

    return xSmooth, ySmooth


def Savgol(data):
    size = (np.ceil(data.size/4) // 2) * 2 + 1
    return savgol_filter(data, int(size), 1)


def Envelope(xRaw, yRaw, xSmooth, ySmooth):
    indices = np.argwhere(np.in1d(xSmooth, xRaw))[:, 0]
    ySmoothMatch = ySmooth[indices]
    diffs = yRaw-ySmoothMatch
    maxOffset = max(diffs)
    minOffset = min(diffs)
    return ySmooth+minOffset, ySmooth+maxOffset


def DoPlotScatterEnvelope(lab, x, y, c, z):
    smoothX, smoothY = SmoothCurve(x, y, 10)
    savX, savY = SmoothCurve(x, Savgol(y), 10)
    plt.plot(smoothX, smoothY, '-', color=c, label="_nolabel", zorder=z, linewidth=2)
    plt.plot(savX, savY, '-', color=list(c) + [0.2], label="_nolabel", zorder=z, linewidth=5)
    plt.scatter(x, y, marker='o', s=30, facecolor="white", edgecolor=c, label=lab, zorder=z)


plt.rcParams['svg.fonttype'] = 'none'
areasAutomated = np.loadtxt(r"figuresAndStats\trackingFigure\data\areasAutomated.dat")

numFrames = areasAutomated.shape[0]

frames = [i * 2 for i in range(numFrames)]

idsToHighlight = {0: (0, 154, 222),
                  1: (255, 198, 30),
                  3: (175, 88, 186),
                  33:(0, 205, 108)}
idsToHighlight = {x: [c / 255 for c in idsToHighlight[x]] for x in idsToHighlight}

print(areasAutomated.shape)
for organoidID in range(areasAutomated.shape[0]):
    if organoidID in idsToHighlight:
        color = idsToHighlight[organoidID]
        zIndex = 2
        label = "#%d" % organoidID
    else:
        color = (0.9, 0.9, 0.9)
        zIndex = 1
        label = "_nolabel"
        continue
    ind = ~np.isnan(areasAutomated[:, organoidID])
    times = np.asarray(frames)[ind]
    areas = areasAutomated[ind, organoidID]
    DoPlotScatterEnvelope(label, times, areas, color, zIndex)

plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel(r"Organoid Area (x $10^3 \mu m^2$)")
plt.xlim([frames[0] - 1, frames[-1] - 1])
plt.show()
