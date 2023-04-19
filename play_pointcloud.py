import matplotlib.pyplot as plt
import json
from tqdm import trange
import os

length_list = list()
xs = list()
ys = list()
zs = list()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

PLOT_RANGE_IN_CM = 500


def plot_3d_scatter(detected_object):
    global xs, ys, zs
    if len(length_list) >= 10:  # clear every X*0.04 s
        xs = xs[length_list[0]:]
        ys = ys[length_list[0]:]
        zs = zs[length_list[0]:]
        length_list.pop(0)
    ax.cla()
    length_list.append(detected_object["NumObj"])
    xs += detected_object["x"]
    ys += detected_object["y"]
    zs += detected_object["z"]
    ax.scatter(xs, ys, zs, c='r', marker='o', label="Radar Data")
    ax.set_xlabel('azimuth (cm)')
    ax.set_ylabel('range (cm)')
    ax.set_zlabel('elevation (cm)')
    ax.set_xlim(-PLOT_RANGE_IN_CM/2, PLOT_RANGE_IN_CM/2)
    ax.set_ylim(0, PLOT_RANGE_IN_CM)
    ax.set_zlim(-PLOT_RANGE_IN_CM/2, PLOT_RANGE_IN_CM/2)
    plt.draw()


files = os.listdir("./output_file")
for i in range(len(files)):
    print(f"{i} >{files[i]}")

choice = int(input("file number:"))

rows = json.load(open(f"output_file/{files[choice]}", 'r'))

for i in trange(len(rows) - 1):
    plot_3d_scatter(rows[i][1])
    plt.pause(rows[i+1][0] - rows[i][0])
