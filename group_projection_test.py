import json
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy


# import time
# import os
# import pprint


def stack_frames(frame):
    return np.dstack(frame)


def calculate_entropy(stacked_frames):
    entropy_map = np.apply_along_axis(lambda x: entropy(x, base=2), axis=2, arr=stacked_frames)
    return entropy_map


def mse(a, b):
    err = np.sum((a - b) ** 2)
    err /= bins ** 2
    return err


def plot_heatmap(entropy_map):
    plt.cla()
    plt.imshow(entropy_map, interpolation='nearest')
    plt.draw()


def calculate_pmf(x):
    total = len(x)
    set_x = list(set(x))
    pmf = [x.count(item) / total for item in set_x]
    return pmf


data = json.load(open("output_file/wei_lr2.json", 'r'))
entropy_list_x = list()
entropy_list_y = list()
delay = 15
bins = 25
prev = np.ndarray((bins, bins))
for i in range(len(data)):
    start_index = i - delay if i > delay else 0
    xs = list()
    ys = list()
    list_of_frames = list()
    for j in range(start_index, i):
        for g in data[i][1]["group"]:
            group = np.array(g).T
            xs = list(group[0])
            ys = list(group[1])

            h, q_x, q_y = np.histogram2d(xs, ys, bins=bins, range=np.array([(-3, 3), (-3, 3)]))
            ssim(h, prev, data_range=np.amax(h) - np.amin(h))
            if mse(h, prev) > 2:
                # print(m)
                h = np.flip(h, 1)
            prev = h
            list_of_frames.append(h)
    if len(list_of_frames) < 15:
        continue
    else:
        list_of_frames = list_of_frames[-15:]
    stacked_frames = stack_frames(list_of_frames)
    entropy_map = calculate_entropy(stacked_frames)
    plot_heatmap(entropy_map)
    # pmf = calculate_pmf(entropy_map)
    # plt.pcolormesh(q_x, q_y, h.T)

    # plt.draw()
    plt.pause(1 / 10)
