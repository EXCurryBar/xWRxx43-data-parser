import matplotlib.pyplot as plt
import json
import os
from tqdm import trange
import numpy as np

accumulated = np.zeros((16,128))
fig = plt.figure()

def plot_range_doppler(heatmap_data):
    plt.clf()
    accumulated = heatmap_data["range-doppler"]*0.7 + accumulated*0.3
    plot_data = heatmap_data["range-doppler"] - accumulated
    cs = plt.contourf(
        np.array(heatmap_data["range-array"], dtype="float32"),
        np.array(heatmap_data["doppler-array"], dtype="float32"),
        np.array(plot_data, dtype="float32"),
        cmap='turbo',
        vmax=1000,
        vmin=0
    )
    fig.colorbar(cs)
    fig.canvas.draw()


if __name__ == "__main__":
    
    files = os.listdir("./output_file")
    for i in range(len(files)):
        print(f"{i} >{files[i]}")

    choice = int(input("file number:"))

    rows = json.load(open(f"output_file/{files[choice]}", 'r'))
    print(rows)
    for i in trange(len(rows) - 1):
        plot_range_doppler(rows[i][1])
        plt.pause(rows[i+1][0] - rows[i][0])

