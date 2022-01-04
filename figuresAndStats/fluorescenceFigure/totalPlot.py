import matplotlib.pyplot as plt
import colorsys
import dill
import numpy as np

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
meanColor = [x / 255 for x in (0, 154, 222)]
lodColor = [x / 255 for x in (255, 31, 91)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
dosagesToUse = [0, 3, 10, 30, 100, 300, 1000]
hue = 343 / 360
colors = [colorsys.hsv_to_rgb(hue, sat, 1) for sat in np.linspace(0.1, 1, 4)] + \
         [colorsys.hsv_to_rgb(hue, 1, val) for val in np.linspace(0.8, 0, 3)]

fluorescenceByDosageA = dill.load(open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceTotalA.pkl", "rb"))
fluorescenceByDosageB = dill.load(open(r"figuresAndStats\fluorescenceFigure\data\fluorescenceTotalB.pkl", "rb"))
areas_by_dosageA = dill.load(open(r"figuresAndStats\fluorescenceFigure\data\areaTotalA.pkl", "rb"))
areas_by_dosageB = dill.load(open(r"figuresAndStats\fluorescenceFigure\data\areaTotalB.pkl", "rb"))
numbers_by_dosageA = dill.load(open(r"figuresAndStats\fluorescenceFigure\data\numberTotalA.pkl", "rb"))
numbers_by_dosageB = dill.load(open(r"figuresAndStats\fluorescenceFigure\data\numberTotalB.pkl", "rb"))


def Plot(fluorescenceByDosage, areas_by_dosage, numbers_by_dosage):
    fig, axes = plt.subplots(4, 1, sharex=True)
    for color, dosage in zip(colors[1:], dosagesToUse[1:]):
        label = "%d nM" % dosage
        fluorescences = np.asarray(fluorescenceByDosage[dosage]) / np.asarray(fluorescenceByDosage[0])
        areas = np.asarray(areas_by_dosage[dosage]) / np.asarray(areas_by_dosage[0])
        numbers = np.asarray(numbers_by_dosage[dosage]) / np.asarray(numbers_by_dosage[0])

        fluorescencePerArea = fluorescences / areas

        fluorescences = fluorescences / (fluorescences[:, 0][:, None])
        areas = areas / (areas[:, 0][:, None])
        numbers = numbers / (numbers[:, 0][:, None])
        fluorescencePerArea = fluorescencePerArea / (fluorescencePerArea[:, 0][:, None])

        axes[0].errorbar(np.arange(0, 73, 4), np.mean(areas, axis=0),
                         yerr=np.std(areas, axis=0) / np.sqrt(areas.shape[0]),
                         label=label, color=color)

        areas = np.asarray(areas_by_dosage[dosage])
        axes[1].errorbar(np.arange(0, 73, 4), np.mean(numbers, axis=0),
                         yerr=np.std(numbers, axis=0) / np.sqrt(areas.shape[0]),
                         label=label, color=color)

        axes[2].errorbar(np.arange(0, 73, 4), np.mean(fluorescences, axis=0),
                         yerr=np.std(fluorescences, axis=0) / np.sqrt(fluorescences.shape[0]),
                         label=label, color=color)

        axes[3].errorbar(np.arange(0, 73, 4), np.mean(fluorescencePerArea, axis=0),
                         yerr=np.std(fluorescencePerArea, axis=0) / np.sqrt(fluorescencePerArea.shape[0]),
                         label=label, color=color)

    axes[0].set_ylabel("Organoid area\n(normalized fold change)")
    axes[1].set_ylabel("Number of organoids\n(normalized fold change)")
    axes[2].set_ylabel("Fluorescence intensity\n(normalized fold change)")
    axes[3].set_ylabel("Fluorescence intensity per area\n(normalized fold change)")
    plt.legend()
    plt.xlabel("Time (hours)")


Plot(fluorescenceByDosageB, areas_by_dosageB, numbers_by_dosageB)
plt.show()
