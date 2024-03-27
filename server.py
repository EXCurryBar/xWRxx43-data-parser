import time
from lib.thread_radar import RadarThread


CLI_BAUD = 115200
DATA_BAUD = 921600

if __name__ == '__main__':
    # Create and start the radar thread
    radar_thread = RadarThread("area_scanner_68xx_ODS.cfg", CLI_BAUD, DATA_BAUD)
    radar_thread.start()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            # To stop the thread and close the connection
            radar_thread.stop()
