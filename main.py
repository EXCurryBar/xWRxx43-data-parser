from lib.radar import Radar
import serial
import time

CLI_BAUD = 115200
DATA_BAUD = 921600


if __name__ == '__main__':
    radar = Radar("1443_30fps_5m_0.049res.cfg", CLI_BAUD, DATA_BAUD)
    while True:
        try:
            dataOK, frameNumber, detObj = radar.parse_data()
            if dataOK:
                # radar.plot_3d_scatter(detObj)
                radar.write_to_json(detObj)
            # let radar rest or whatever
            time.sleep(1/30)

        except KeyboardInterrupt or serial.SerialException:
            # if ^C pressed
            radar.close_connection()
            print("\nPeace")
            break
