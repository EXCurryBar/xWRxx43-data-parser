import json
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open("output_file/test1.json", 'r'))
entropy_list_x = list()
entropy_list_y = list()
delay = 15
bins = 25
# prev = np.ndarray((bins, bins))
list_of_entropy = list()
list_of_x = list()
list_of_y = list()
list_of_atan = list()
for i in range(len(data)):
    start_index = i - delay if i > delay else 0
    xs = list()
    ys = list()

    # for _ in range(start_index, i):
    for g in data[i][1].get("group", []):
        group = np.array(g).T
        xs += list(group[0])
        ys += list(group[1])
    h, q_x, q_y = np.histogram2d(xs, ys, bins=bins, range=np.array([(-3, 3), (-3, 3)]))
    y_0 = 0
    x_0 = 0
    for row in h:
        if all(item <= 10 for item in row):
            x_0 += 1

    x_percentage = (bins - x_0) / bins

    for row in h.T:
        if all(item <= 10 for item in row):
            y_0 += 1

    y_percentage = (bins - y_0) / bins
    list_of_x.append(x_percentage)
    list_of_y.append(y_percentage)
    list_of_entropy.append([x_percentage, y_percentage])
    try:
        at = np.arctan(y_percentage / x_percentage)
    except ZeroDivisionError:
        at = 0
    if np.pi < at * 8 < 3 * np.pi and y_percentage >= 0.3 and x_percentage >= 0.3:
        print(at * 180 / np.pi)
        print(y_percentage, x_percentage)
        print("FALL")
        plt.pause(0)
    plt.figure(1)
    plt.pcolormesh(q_x, q_y, h.T)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.pause(1 / 10)
