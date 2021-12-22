import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'
performance = np.loadtxt(r"figuresAndStats\trackingFigure\data\performance.dat")
areasGT = np.loadtxt(r"figuresAndStats\trackingFigure\data\areasGT.dat")
areasAutomated = np.loadtxt(r"figuresAndStats\trackingFigure\data\areasAutomated.dat")

correctPerFrame = performance[0, :]
incorrectPerFrame = performance[1, :]
percentCorrect = performance[2, :]

numFrames = performance.shape[1]

frames = [i * 2 for i in range(numFrames)]

idsToHighlight = {0: (0, 154, 222),
                  1: (255, 198, 30),
                  3: (175, 88, 186),
                  33: (255, 31, 91)}
idsToHighlight = {x: [c / 255 for c in idsToHighlight[x]] for x in idsToHighlight}
# plt.subplot(2, 1, 2)
plt.plot(frames, np.delete(areasAutomated, list(idsToHighlight.keys()), 1), '-', color=(0.9, 0.9, 0.9),
         label="_nolabel")
backgroundIDs = np.delete(areasAutomated, list(idsToHighlight.keys()), 1)
repFrames = np.repeat(np.asarray(frames)[:, None], backgroundIDs.shape[1], 1)
print(backgroundIDs.shape)
print(repFrames.shape)
plt.scatter(repFrames, backgroundIDs, marker='o', color=(0.9, 0.9, 0.9),
            label="_nolabel", s=10)
[plt.plot(frames, areasAutomated[:, idToHighlight], '-', color=idsToHighlight[idToHighlight], label=str(idToHighlight))
 for idToHighlight in idsToHighlight]
[plt.scatter(frames, areasAutomated[:, idToHighlight], marker='o', color=idsToHighlight[idToHighlight],
             label="_nolabel", s=10)
 for idToHighlight in idsToHighlight]
plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel(r"Organoid Area (x $10^3 \mu m^2$)")
plt.xlim([frames[0] - 1, frames[-1] - 1])
plt.show()
