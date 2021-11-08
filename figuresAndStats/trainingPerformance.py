import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import numpy as np

lines = open(r"figuresAndStats\trainingLosses.csv", "r").read().split("\n")[1:-1]
losses = np.asarray([[float(x) for x in line.split(",")] for line in lines])
print(losses.shape)
epoch = losses[:, 0]
trainingLosses = losses[:, 1]
validationLosses = losses[:, 2]

plt.plot(trainingLosses, "r")
plt.plot(validationLosses, "b")
plt.plot(trainingLosses, "or")
plt.plot(validationLosses, "ob")
plt.legend(["Training set", "Validation set"])
plt.xlabel("Epoch")
plt.ylabel("Loss (binary cross-entropy)")
plt.title("OrganoID neural network training performance")
plt.show()
