import time
import os
import traceback
from lib.radar import Radar
from serial import SerialException

CLI_BAUD = 115200
DATA_BAUD = 921600

if __name__ == '__main__':
    os.makedirs("./output_file", exist_ok=True)
    os.makedirs("./raw_file", exist_ok=True)
    radar = Radar("area_scanner_68xx_ODS.cfg", CLI_BAUD, DATA_BAUD, remove_static_noise=False, write_file=False)
    while True:
        try:
            data_ok, frame_number, radar_data = radar.parse_data()
            if data_ok:
                radar.plot_3d_scatter(radar_data)
                # radar.process_cluster(radar_data, thr=30, delay=15)
                # time.sleep(1/10)
                continue
        except KeyboardInterrupt or SerialException:
            # if ^C pressed
            radar.close_connection()
            print("\nPeace")
            break

        except Exception as e:
            radar.close_connection()
            print(f"\nShit code: {e}")
            print(traceback.print_exc())
            break
