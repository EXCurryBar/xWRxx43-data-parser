import os
import sys

while True:
    try:
        choice = int(input("Radar Startup Script:\n\t1 > main.py\n\t2 > collect_data.py\ninput:"))
        if choice in [1, 2]:
            break
        else:
            print("Please enter valid option")
    except ValueError:
        print("Please enter valid option")
if sys.platform == "linux":
    os.system("sudo chmod 666 /dev/ttyUSB0")
    os.system("sudo chmod 666 /dev/ttyUSB1")

if choice == 1:
    from main import main
else:
    from collect_data import main

main()
