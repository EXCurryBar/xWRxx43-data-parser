import matplotlib.pyplot as plt
import json
from tqdm import trange
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, fcluster

length_list = list()
xs = list()
ys = list()
zs = list()
vs = list()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

PLOT_RANGE_IN_METER = 5
RADAR_HEIGHT_IN_METER = 1.83

def process_cluster(detected_object, thr=10, delay=10):
    global xs, ys, zs, vs, length_list
    data = dict()
    points = detected_object["scatter"]
    print(points)
    if len(length_list) >= delay:  # delay x * 0.1 s
        xs = xs[length_list[0]:]
        ys = ys[length_list[0]:]
        zs = zs[length_list[0]:]
        vs = vs[length_list[0]:]
        length_list.pop(0)
    length_list.append(len(points["x"]))

    xs += list(points["x"])
    ys += list(points["y"])
    zs += list(points["z"])
    vs += list(points["v"])
    scatter_data = np.array([item for item in zip(xs, ys, zs, vs)])
    if len(xs) > thr:
        try:
            Z = linkage(scatter_data, method="complete", metric="euclidean")
        except:
            return 'r', [], []
        clusters = fcluster(Z, 1.6, criterion='distance')
        color = list(clusters)
        labels = set(color)
        bounding_boxes = list()
        groups = list()
        for label in labels:
            xs = list()
            ys = list()
            zs = list()
            vs = list()
            if color.count(label) < thr:
                outlier_index = [i for i in range(len(xs)) if color[i] == label]
                for index in sorted(outlier_index, reverse=True):
                    del xs[index]
                    del ys[index]
                    del zs[index]
                    del vs[index]
                    del color[index]
                continue
            xs.append([xs[i] for i in range(len(xs)) if color[i] == label])
            ys.append([ys[i] for i in range(len(ys)) if color[i] == label])
            zs.append([zs[i] for i in range(len(zs)) if color[i] == label])
            vs.append([vs[i] for i in range(len(vs)) if color[i] == label])
            x1 = -999
            y1 = -999
            z1 = -999

            x2 = 999
            y2 = 999
            z2 = 999
            group = [xs, ys, zs, vs]
            for idx, value in enumerate(color):
                if value == label:
                    x, y, z = scatter_data[idx][:3]
                    x1 = x if x > x1 else x1
                    y1 = y if y > y1 else y1
                    z1 = z if z > z1 else z1

                    x2 = x if x2 > x else x2
                    y2 = y if y2 > y else y2
                    z2 = z if x2 > z else z2
                    
                    group.append(scatter_data[idx][:3])
                    # print(x1, y1, z1)
                    # print(x2, y2, z2)
            groups.append(group)
            z2 = -RADAR_HEIGHT_IN_METER if z2 == 999 else z2
            bounding_boxes.append(
                [
                    [[x1, y1, z1], [x1, y2, z1]],
                    [[x1, y1, z1], [x2, y1, z1]],
                    [[x1, y1, z1], [x1, y1, z2]],
                    [[x1, y1, z2], [x1, y2, z2]],
                    [[x1, y1, z2], [x2, y1, z2]],
                    [[x2, y1, z2], [x2, y2, z2]],
                    [[x2, y1, z2], [x2, y1, z1]],
                    [[x2, y1, z1], [x2, y2, z1]],
                    [[x2, y2, z2], [x2, y2, z1]],
                    [[x2, y2, z2], [x1, y2, z2]],
                    [[x1, y2, z2], [x1, y2, z1]],
                    [[x1, y2, z1], [x2, y2, z1]]
                ]
            )
            data.update({
                "scatter":points,
                "bounding_box":bounding_boxes,
                "group": groups,
                "label": color,
            })
        return color, groups, bounding_boxes
    return 'r', [], []

def plot_3d_scatter(detected_object):
        global xs, ys, zs, vs, length_list
        label, groups, bounding_boxes = process_cluster(detected_object, thr=10, delay=15)
        ax.cla()
        if bounding_boxes:
            # print("#################################")
            for box in bounding_boxes:
                # print("----------------------------")
                for line in box:
                    vertex1 = line[0]
                    vertex2 = line[1]

                    edge_x = [vertex1[0], vertex2[0]]
                    edge_y = [vertex1[1], vertex2[1]]
                    edge_z = [vertex1[2], vertex2[2]]

                    plt.plot(edge_x, edge_y, edge_z, c='g', marker=None, linestyle='-', linewidth=2)

        # center_x = tracker["x"]
        # center_y = tracker["y"]
        # center_z = tracker["z"]

        ax.scatter(xs, ys, zs, c=label, marker='o', label="Radar Data")
        # ax.scatter(center_x, center_y, center_z, s=8**2, c='g', marker='^', label="Center Points")
        # print(diff_xyz)
        ax.set_xlabel('X(m)')
        ax.set_ylabel('range (m)')
        ax.set_zlabel('elevation (m)')
        ax.set_xlim(-PLOT_RANGE_IN_METER/2, PLOT_RANGE_IN_METER/2)
        ax.set_ylim(0, PLOT_RANGE_IN_METER)
        ax.set_zlim(-RADAR_HEIGHT_IN_METER, RADAR_HEIGHT_IN_METER)
        plt.draw()



files = os.listdir("./output_file")
for i in range(len(files)):
    print(f"{i} >{files[i]}")

choice = int(input("file number:"))

rows = json.load(open(f"output_file/{files[choice]}", 'r'))

for i in trange(len(rows) - 1):
    plot_3d_scatter(rows[i][1])
    plt.pause(rows[i+1][0] - rows[i][0])
