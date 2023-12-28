import json
import matplotlib.pyplot as plt
import numpy as np


data = json.load(open("output_file/wei_lr1.json", 'r'))
thetas = list()
phis = list()

for i, row in enumerate(data):
    time = row[0]
    # groups = row[1]["group"]
    # vectors = row[1]["vector"]
    try:
        values = row[1]["eigenvalues"]
    except KeyError:
        continue
    plt.cla()
    for value in values:
        theta = np.arctan(np.sqrt(value[0]**2 + value[1]**2) / value[2])
        phi = np.arctan(value[1] / value[0])
        thetas.append(theta)
        phis .append(phi)

        plt.scatter(thetas, phis)
        plt.ylim((-2, 2))
        plt.xlim((-2, 2))
        plt.draw()
        plt.pause(1/10)
        thetas.clear()
        phis.clear()
