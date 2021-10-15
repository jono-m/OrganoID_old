import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from util.stats import pearsonr_ci, linr_ci

csvFile = open(r"C:\Users\jonoj\Documents\ML\TrackingResults\track_2021_10_13_11_18_37\track_2021_10_13_11_18_37.csv",
               "r")
data = csvFile.read()
organoids = data.split("\n")[1:-1]
organoids = [organoid.split(", ")[1:] for organoid in organoids]


def convertToArea(organoidAreas):
    a = []
    for x in organoidAreas:
        if x == '':
            a.append(np.nan)
        elif x == 'None':
            a.append(a[-1])
        else:
            a.append(int(x))
    return a


imageWidthInMicrons = 1.31 * 1024
pixelWidthInMicrons = imageWidthInMicrons / 512
areaPerPixel = pixelWidthInMicrons * pixelWidthInMicrons

areas = [convertToArea(x) for x in organoids]
focusOrganoids = np.asarray([1, 2, 3, 4, 6, 7, 10, 19, 29])
npAreas = np.asarray(areas) * areaPerPixel
focusAreas = npAreas[focusOrganoids - 1]
otherAreas = np.delete(npAreas, focusOrganoids - 1, 0)
hours = np.arange(len(areas[0])) * 2
plt.plot(hours, otherAreas.transpose(), color='0.9', label='_nolegend_')
plt.plot(hours, focusAreas.transpose(), 'o-')
plt.legend(focusOrganoids)
plt.xlabel("Hours")
plt.ylabel(r"Area ($\mu m^2$)")
# plt.plot(npAreas.transpose(), color='0.9')
plt.show()
plt.savefig(r"C:\Users\jonoj\Downloads\test.fig")
