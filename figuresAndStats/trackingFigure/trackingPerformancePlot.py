import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['svg.fonttype'] = 'none'
performance = np.loadtxt(r"figuresAndStats\trackingFigure\data\performance.dat")
a = [x / 255 for x in (0, 205, 108)]
plt.plot(np.arange(performance.shape[1]) * 4, performance[2, :], color=a)
plt.ylim([0, 1.1])
plt.xlabel("Time (hours)")
plt.ylabel("Tracking accuracy")
plt.axhline(y=min(performance[2, :]), color="black", linestyle="--")
print(min(performance[2, :]))
plt.show()
