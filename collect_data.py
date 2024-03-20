import multiprocessing
import time
import os
import numpy as np
import pyaudio
import traceback
from lib.radar import Radar

ev = multiprocessing.Event()

CLI_BAUD = 115200
DATA_BAUD = 921600
p = pyaudio.PyAudio()
volume = 0.5
fs = 44100
duration = 1.0
f = 440.0
samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
output_bytes = (volume * samples).tobytes()


def initialize_radar(name=None):
    os.makedirs("./output_file", exist_ok=True)
    os.makedirs("./raw_file", exist_ok=True)
    return Radar("area_scanner_68xx_ODS.cfg", CLI_BAUD, DATA_BAUD, remove_static_noise=False, write_file=True,
                 file_name=name)


def beep():
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    time.sleep(3)
    stream.write(output_bytes)
    # stream.stop_stream()
    # stream.close()
    time.sleep(5)
    stream.write(output_bytes)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return


def collect_data(se, name):
    radar = initialize_radar(name)
    th = multiprocessing.Process(target=beep, args=())
    th.start()
    while th.is_alive():
        try:
            data_ok, frame_number, radar_data = radar.parse_data()
            if data_ok:
                s = time.time()
                radar.process_cluster(radar_data, thr=30, delay=15)
                delay = time.time() - s
                time.sleep(1 / 10 - delay)
                continue

        except Exception as e:
            radar.close_connection()
            print(f"\nShit code: {e}")
            print(traceback.print_exc())
            break

    radar.close_connection()
    return


ACTION = ["light_fall_lr", "light_fall_rl", "light_fall_fw", "light_fall_bw", "heavy_fall_lr", "heavy_fall_rl",
          "heavy_fall_fw", "heavy_fall_bw", "walk_fall_lr", "walk_fall_rl", "walk_fall_fw", "walk_fall_bw",
          "sit", "squat", "pick_thing", "walk_pass_fw", "walk_pass_bw", "walk_pass_lr", "walk_pass_rl", "making_bed",
          "swing", "step_still", "rotate_hand", "raise_hand"]
SET = 3


def main():
    subject = input("Enter subject name: ")
    for i, action in enumerate(ACTION):
        for j in range(SET):
            input(f"enter to start collecting {subject}'s {action} no.{j}:")
            t1 = multiprocessing.Process(target=collect_data, args=(ev, f"{subject}_{action}_{j}",))
            t1.start()

            ev.set()
            t1.join()
            ev.clear()


if __name__ == '__main__':
    # main()
    collect_data(None, "test1")
    # beep()
