import numpy as np
import matplotlib.pyplot as plt
import pandas

plt.rcParams['svg.fonttype'] = 'none'

data = pandas.read_excel(r"Z:\ML_Organoid\Paper\Data\Raw Figure Data.xlsx", sheet_name="Figure 3j",
                         usecols=np.arange(40), index_col=0).to_numpy()


def convertToArea(organoidAreas):
    a = []
    for x in organoidAreas:
        if x == ' ':
            a.append(np.nan)
        elif x == 'MISSING':
            a.append(a[-1])
        else:
            a.append(int(x))
    return a


imageWidthInMicrons = 1.31 * 1024
pixelWidthInMicrons = imageWidthInMicrons / 512
areaPerPixel = pixelWidthInMicrons * pixelWidthInMicrons

areas = [convertToArea(list(data[:, x])) for x in range(data.shape[1])]
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
plt.show()
