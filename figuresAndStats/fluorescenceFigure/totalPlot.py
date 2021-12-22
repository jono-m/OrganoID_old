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
    for color, dosage in zip(colors, dosagesToUse):
        label = "%d nM" % dosage
        fluorescences = np.asarray(fluorescenceByDosage[dosage])
        areas = np.asarray(areas_by_dosage[dosage])
        numbers = np.asarray(numbers_by_dosage[dosage])
        plt.subplot(2, 2, 1)
        plt.errorbar(np.arange(0, 73, 4), np.mean(areas, axis=0),
                     yerr=np.std(areas, axis=0) / np.sqrt(areas.shape[0]),
                     label=label, color=color)

        areas = np.asarray(areas_by_dosage[dosage])
        plt.subplot(2, 2, 2)
        plt.errorbar(np.arange(0, 73, 4), np.mean(numbers, axis=0),
                     yerr=np.std(numbers, axis=0) / np.sqrt(areas.shape[0]),
                     label=label, color=color)

        plt.subplot(2, 2, 3)
        plt.errorbar(np.arange(0, 73, 4), np.mean(fluorescences, axis=0),
                     yerr=np.std(fluorescences, axis=0) / np.sqrt(fluorescences.shape[0]),
                     label=label, color=color)

        plt.subplot(2, 2, 4)
        fluorescencePerArea = np.divide(fluorescences, areas)
        plt.errorbar(np.arange(0, 73, 4), np.mean(fluorescencePerArea, axis=0),
                     yerr=np.std(fluorescencePerArea, axis=0) / np.sqrt(fluorescencePerArea.shape[0]),
                     label=label, color=color)

    plt.subplot(2, 2, 1)
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Organoid area (fold change from t=0)")

    plt.subplot(2, 2, 2)
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Number of organoids (fold change from t=0)")

    plt.subplot(2, 2, 3)
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Fluorescent Death Signal (fold change from t=0)")

    plt.subplot(2, 2, 4)
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Fluorescent Death Signal per Area (fold change from t=0)")


Plot(fluorescenceByDosageA, areas_by_dosageA, numbers_by_dosageA)
plt.figure()
Plot(fluorescenceByDosageB, areas_by_dosageB, numbers_by_dosageB)
plt.show()