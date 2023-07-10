import matplotlib.pyplot as plt
import json
import os
from tqdm import trange
import numpy as np

prev = dict()
fig = plt.figure()


def plot_range_doppler(heatmap_data):
    plt.clf()
    try:
        cs = plt.contourf(
            heatmap_data["range-array"],
            heatmap_data["doppler-array"],
            heatmap_data["range-doppler"],
            # vmax=1000,
            # vmin=200
        )
        fig.colorbar(cs)
        fig.canvas.draw()
    except KeyError:
        pass


if __name__ == "__main__":
    
    files = os.listdir("./output_file")
    for i in range(len(files)):
        print(f"{i} >{files[i]}")

    choice = int(input("file number:"))

    rows = json.load(open(f"output_file/{files[choice]}", 'r'))
    for i in trange(len(rows) - 1):
        if len(rows[i][1]["range_doppler"]) == 0:
            continue
        plot_range_doppler(rows[i][1]["range_doppler"])
        plt.pause(rows[i+1][0] - rows[i][0])

