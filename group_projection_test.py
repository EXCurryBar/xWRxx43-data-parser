import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
# import time
# import os
# import pprint


def calculate_pmf(x):
    total = len(x)
    set_x = list(set(x))
    pmf = [x.count(item) / total for item in set_x]
    return pmf


data = json.load(open("output_file/wei_lr1.json", 'r'))
entropy_list_x = list()
entropy_list_y = list()
delay = 15
bins = 25

for i in range(len(data)):
    start_index = i - delay if i > delay else 0
    xs = list()
    ys = list()
    plt.cla()
    for j in range(start_index, i):
        for g in data[i][1]["group"]:
            group = np.array(g).T
            xs += list(group[0])
            ys += list(group[1])
    h, q_x, q_y = np.histogram2d(xs, ys, bins=bins, range=np.array([(-3, 3), (-3, 3)]))
    print(np.shape(h), np.shape(q_x), np.shape(q_y))
    print(np.max(h), np.min(h))
    plt.pcolormesh(q_x, q_y, h.T)

    plt.draw()
    plt.pause(1 / 10)
