import pprint

from lib.radar import Radar
from serial import SerialException
from time import sleep

CLI_BAUD = 115200
DATA_BAUD = 921600


if __name__ == '__main__':
    radar = Radar("heatmap.cfg", CLI_BAUD, DATA_BAUD)
    while True:
        try:
            data_ok, frame_number, detected_object, range_doppler_data, range_profile = radar.parse_data()
            if data_ok:
                # radar.write_to_json(detObj, range_bin)
                radar.plot_heat_map(detected_object)
            # sleep(1/5)
        except KeyboardInterrupt or SerialException:
            # if ^C pressed
            radar.close_connection()
            print("\nPeace")
            break

        except Exception as e:
            radar.close_connection()
            print(f"\nShit code: {e}")
            break
