import json
import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import entropy


# def stack_frames(frame):
#     return np.dstack(frame)


# def calculate_entropy(stacked_frames):
#     entropy_map = np.apply_along_axis(lambda x: entropy(x, base=10), axis=2, arr=stacked_frames)
#     return entropy_map


# def mse(a, b):
#     err = np.sum((a - b) ** 2)
#     err /= bins ** 2
#     return err


# def plot_heatmap(entropy_map):
#     plt.cla()
#     plt.imshow(entropy_map, interpolation='nearest')
#     plt.draw()


# def calculate_pmf(x):
#     total = len(x)
#     set_x = list(set(x))
#     pmf = [x.count(item) / total for item in set_x]
#     return pmf


data = json.load(open("output_file/wei_forward.json", 'r'))
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
    for g in data[i][1]["group"]:
        group = np.array(g).T
        xs += list(group[0])
        ys += list(group[1])
    h, q_x, q_y = np.histogram2d(xs, ys, bins=bins, range=np.array([(-3, 3), (-3, 3)]))
    y_0 = 0
    x_0 = 0
    for row in h:
        if all(item <= 0.2 for item in row):
            x_0 += 1

    x_percentage = x_0 / bins

    for row in h.T:
        if all(item <= 0.2 for item in row):
            y_0 += 1

    y_percentage = y_0 / bins
    # list_of_frames.append(h)
    list_of_x.append(x_percentage)
    list_of_y.append(y_percentage)
    list_of_entropy.append([x_percentage, y_percentage])
    # print(list_of_entropy)
    list_of_atan.append(at := np.arctan(y_percentage / x_percentage))
    if 44 < at*180/np.pi < 50:
        print("FALL")
    # print(y_0, x_0)
    plt.figure(1)
    # plt.plot(list_of_atan, c='r')
    plt.pcolormesh(q_x, q_y, h.T)
    # plt.figure(2)
    # plt.scatter(list_of_x, list_of_y, c="r")
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    plt.pause(0)
    # stacked_frames = stack_frames(list_of_frames)
    # entropy_map = calculate_entropy(stacked_frames)
    # plot_heatmap(entropy_map)
# plt.pause(0)
