import pprint

from lib.radar import Radar
from serial import SerialException
import matplotlib.pyplot as plt

CLI_BAUD = 115200
DATA_BAUD = 921600


if __name__ == '__main__':
    radar = Radar("rangedoppler.cfg", CLI_BAUD, DATA_BAUD)
    while True:
        try:
            dataOK, frameNumber, range_doppler = radar.parse_data()
            if dataOK:
                pprint.pprint(range_doppler)
                # detObj = radar.remove_static(detObj)
                # radar.write_to_json(detObj)

        except KeyboardInterrupt or SerialException:
            # if ^C pressed
            radar.close_connection()
            print("\nPeace")
            break

        except Exception as e:
            radar.close_connection()
            print(f"\nShit code: {e}")
            break
