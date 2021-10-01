import matplotlib.pyplot as plt
import numpy as np

trainingData = open(r"C:\Users\jonoj\Documents\ML\trainingLoss.csv", "r").read().split(",")
trainingLosses = np.asarray([float(x) for x in trainingData])

validationData = open(r"C:\Users\jonoj\Documents\ML\validationLoss2.csv", "r").read().split(",")
validationLosses = np.asarray([float(x) for x in validationData])

plt.plot(trainingLosses, "r")
plt.plot(validationLosses, "b")
plt.plot(trainingLosses, "or")
plt.plot(validationLosses, "ob")
plt.legend(["Training set", "Validation set"])
plt.xlabel("Epoch")
plt.ylabel("Loss (binary cross-entropy)")
plt.title("OrganoID neural network training performance")
plt.show()
