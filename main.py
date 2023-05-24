from lib.radar import Radar
from serial import SerialException
import matplotlib.pyplot as plt

CLI_BAUD = 115200
DATA_BAUD = 921600


if __name__ == '__main__':
    radar = Radar("heatmap.cfg", CLI_BAUD, DATA_BAUD)
    while True:
        try:
            dataOK, frameNumber, range_bin, QQ = radar.parse_data()
            if dataOK:
                # radar.write_to_json(detObj, range_bin)
                print(QQ)

        except KeyboardInterrupt or SerialException:
            # if ^C pressed
            radar.close_connection()
            print("\nPeace")
            break

        except Exception as e:
            radar.close_connection()
            print(f"\nShit code: {e}")
            break
