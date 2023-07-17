import time
import traceback
from lib.radar import Radar
from serial import SerialException

CLI_BAUD = 115200
DATA_BAUD = 921600


if __name__ == '__main__':
    radar = Radar("6843_scatter.cfg", CLI_BAUD, DATA_BAUD, remove_static_noise=False, write_file=False)
    while True:
        try:
            data_ok, frame_number, radar_data = radar.parse_data()
            # if data_ok:
                # radar.plot_3d_scatter(radar_data["3d_scatter"])
                # radar.plot_range_doppler(radar_data["range_doppler"])
                # radar.plot_heat_map(radar_data["azimuth_heatmap"])
            time.sleep(1/30)

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
