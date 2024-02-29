import multiprocessing
import time
import os
import traceback
from lib.radar import Radar

ev = multiprocessing.Event()

CLI_BAUD = 115200
DATA_BAUD = 921600


def initialize_radar(name=None):
    os.makedirs("./output_file", exist_ok=True)
    os.makedirs("./raw_file", exist_ok=True)
    return Radar("area_scanner_68xx_ODS.cfg", CLI_BAUD, DATA_BAUD, remove_static_noise=False, write_file=True,
                 file_name=name)


def collect_data(se, name):
    radar = initialize_radar(name)
    while not se.is_set():
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


ACTION = 5
SET = 3


def main():
    subject = input("Enter subject name: ")
    for i in range(ACTION):
        action = input("enter to start collecting data")
        for j in range(SET):
            t1 = multiprocessing.Process(target=collect_data, args=(ev, f"{subject}_{i}_{j}",))
            t1.start()
            input("press enter to stop")

            ev.set()
            t1.join()
            ev.clear()


if __name__ == '__main__':
    main()
