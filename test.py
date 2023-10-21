import json
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pprint

data = json.load(open("output_file/2023-10-19-1541.json", 'r'))

for row in data:
    groups = row[1]["group"]
    plt.cla()
    for group in groups:
        xy = np.array(group).T
        x = xy[0]
        y = xy[1]
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))
        plt.scatter(x, y, marker='o', label="group")

    plt.draw()
    plt.pause(1 / 10)
    # os.system("pause")
