import serial
import serial.tools.list_ports
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Radar:
    def __init__(self, cli_baud_rate=115200, data_baud_rate=921600):
        port = self.read_com_port()
        self._cli = serial.Serial(port["CliPort"], cli_baud_rate)
        self._data = serial.Serial(port["DataPort"], data_baud_rate)
        self._config = list()
        self._config_parameter = dict()

    def send_config(self):
        self._config = open("./radar_config/1443_30fps_5m.cfg").readlines()
        for command in self._config:
            print(command)
            self._cli.write((command + '\n').encode())
            time.sleep(0.01)

    def parse_config(self):
        num_rx = 4
        num_tx = 3
        for line in self._config:
            split_word = line.split(' ')

            if "profileCfg" in split_word[0]:
                start_frequency = int(float(split_word[2]))
                idle_time = int(split_word[3])
                ramp_end_time = float(split_word[5])
                frequency_slope_const = float(split_word[8])
                adc_samples = int(split_word[10])
                adc_samples_next = 1
                while adc_samples > adc_samples_next:
                    adc_samples_next *= 2
                sample_rate = int(split_word[11])

            elif "frameCfg" in split_word[0]:
                chirp_start_index = int(split_word[1])
                chirp_end_index = int(split_word[2])
                loop_count = int(split_word[3])
                frame_count = int(split_word[4])
                frame_periodicity = int(float(split_word[5]))

        chirps_per_frame = (chirp_end_index - chirp_start_index + 1) * loop_count

        self._config_parameter.update(
            {
                "DopplerBins": int(chirps_per_frame/num_tx),
                "RangeBins": int(adc_samples_next),
                "RangeResolution": (3e8*sample_rate*1e3)/(2*frequency_slope_const*1e12*adc_samples),
                "RangeIndexToMeters": (3e8*sample_rate*1e3)/(2*frequency_slope_const*1e12*int(adc_samples_next)),
                "DopplerResolution":
                    3e8/(2*start_frequency*1e9*(idle_time+ramp_end_time)*1e6*int(chirps_per_frame/num_tx)),
                "MaxRange": (300*0.9*sample_rate)/(2*frequency_slope_const*1e3),
                "MaxVelocity": 3e8/(4*start_frequency*1e9*(idle_time+ramp_end_time)*1e6*num_tx)
            }
        )

    def parse_data(self):
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]
        byte_buffer = np.zeros(2**15, dtype='uint8')
        byte_buffer_length = 0

        object_struct_size = 12
        byte_vector_acc_max_size = 2**15
        demo_uart_msg_detected_points = 1
        demo_uart_msg_range_profile = 2
        max_buffer_size = 2**15
        magic_word = [2, 1, 4, 3, 6, 5, 8, 7]

        magic_ok = 0
        data_ok = 0
        frame_number = 0
        detected_object = dict()

        read_buffer = self._data.read(self._data.in_waiting)
        byte_vector = np.frombuffer(read_buffer, dtype='uint8')
        byte_count = len(byte_vector)

        if (byte_buffer_length + byte_count) < max_buffer_size:
            byte_buffer[byte_buffer_length:byte_buffer_length+byte_count] = byte_vector[:byte_count]
            byte_buffer_length += byte_count

        if byte_buffer_length > 16:
            possible_location = np.where(byte_vector == magic_word[0])[0]

            start_index = list()
            for loc in possible_location:
                check = byte_vector[loc:loc+8]
                if np.array_equal(check, magic_word):
                    start_index.append(loc)

            if start_index:
                if start_index[0] > 0:
                    byte_buffer[:byte_buffer_length-start_index[0]] = byte_buffer[start_index[0]:byte_buffer_length]
                    byte_buffer_length -= start_index[0]

                if byte_buffer_length < 0:
                    byte_buffer_length = 0

                total_packet_length = np.matmul(byte_buffer[12:12+4], word)

                if (byte_buffer_length >= total_packet_length) and (byte_buffer_length != 0):
                    magic_ok = 1

            if magic_ok:
                # return True, True, True
                index = 0
                magic_number = byte_buffer[index:index+8]
                index += 8
                version = format(np.matmul(byte_buffer[index:index+4], word), 'x')
                index += 4
                total_packet_length = np.matmul(byte_buffer[index:index+4], word)
                index += 4
                platform = format(np.matmul(byte_buffer[index:index+4], word), 'x')
                index += 4
                frame_number = np.matmul(byte_buffer[index:index+4], word)
                index += 4
                time_cpu_cycle = np.matmul(byte_buffer[index:index+4], word)
                index += 4
                num_detected_object = np.matmul(byte_buffer[index:index+4], word)
                index += 4
                tlvs = np.matmul(byte_buffer[index:index+4], word)
                index += 4

                for _ in range(tlvs):
                    tlv_type = np.matmul(byte_buffer[index:index+4], word)
                    index += 4
                    tlv_length = np.matmul(byte_buffer[index:index+4], word)
                    index += 4
                    if tlv_type == demo_uart_msg_detected_points:
                        tlv_num_obj = np.matmul(byte_buffer[index:index+2], word[:2])
                        index += 2
                        tlv_xyz_format = np.matmul(byte_buffer[index:index+2], word[:2])
                        index += 2

                        range_index = np.zeros(tlv_num_obj, dtype='int16')
                        doppler_index = np.zeros(tlv_num_obj, dtype='int16')
                        peak_value = np.zeros(tlv_num_obj, dtype='int16')
                        x = np.zeros(tlv_num_obj, dtype='int16')
                        y = np.zeros(tlv_num_obj, dtype='int16')
                        z = np.zeros(tlv_num_obj, dtype='int16')

                        for i in range(tlv_num_obj):
                            range_index[i] = np.matmul(byte_buffer[index:index + 2], word[:2])
                            index += 2
                            doppler_index[i] = np.matmul(byte_buffer[index:index + 2], word[:2])
                            index += 2
                            peak_value[i] = np.matmul(byte_buffer[index:index + 2], word[:2])
                            index += 2
                            x[i] = np.matmul(byte_buffer[index:index + 2], word[:2])
                            index += 2
                            y[i] = np.matmul(byte_buffer[index:index + 2], word[:2])
                            index += 2
                            z[i] = np.matmul(byte_buffer[index:index + 2], word[:2])
                            index += 2

                        range_value = range_index * self._config_parameter["RangeIndexToMeters"]
                        doppler_index[doppler_index > (self._config_parameter["DopplerBins"]/2-1)] = \
                            doppler_index[doppler_index > (self._config_parameter["DopplerBins"]/2-1)]-65535
                        doppler_value = doppler_index * self._config_parameter["DopplerResolution"]
                        x = x / tlv_xyz_format
                        y = y / tlv_xyz_format
                        z = z / tlv_xyz_format

                        detected_object.update(
                            {
                                "NumObj": tlv_num_obj,
                                "RangeIndex": range_index,
                                "Range": range_value,
                                "DopplerIndex": doppler_index,
                                "Doppler": doppler_value,
                                "PeakValue": peak_value,
                                "x": x,
                                "y": y,
                                "z": z
                            }
                        )
                        data_ok = 1

                    elif tlv_type == demo_uart_msg_range_profile:
                        index += tlv_length
                if index > 0 and data_ok == 1:
                    shift_index = index
                    byte_buffer[:byte_buffer_length - shift_index] = byte_buffer[shift_index:byte_buffer_length]
                    byte_buffer_length -= shift_index

                    if byte_buffer_length < 0:
                        byte_buffer_length = 0

        return data_ok, frame_number, detected_object

    def close_connection(self):
        self._cli.write("sensorStop\n".encode())
        time.sleep(0.5)
        self._cli.close()
        self._data.close()

    def read_com_port(self):
        data_port = ""
        cli_port = ""
        ports = serial.tools.list_ports.comports(include_links=False)
        for port in ports:
            if "XDS110 Class Auxiliary Data Port" in port.description:
                data_port = port.name
            elif "XDS110 Class Application/User" in port.description:
                cli_port = port.name

        if not data_port or not cli_port:
            input("please connect the radar and press Enter...")
            return self.read_com_port()
        else:
            return {
                "DataPort": data_port,
                "CliPort": cli_port
            }


if __name__ == '__main__':
    length_list = list()
    Xs = list()
    Ys = list()
    Zs = list()
    radar = Radar()
    radar.send_config()
    radar.parse_config()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    while True:
        try:
            dataOK, frameNumber, detObj = radar.parse_data()
            if dataOK:
                if len(length_list) >= 10:  # delay x * 0.04 s
                    count = 0
                    Xs = Xs[length_list[0]:]
                    Ys = Ys[length_list[0]:]
                    Zs = Zs[length_list[0]:]
                    length_list.pop(0)
                ax.cla()
                length_list.append(len(detObj["x"]))
                Xs += list(detObj["x"])
                Ys += list(detObj["y"])
                Zs += list(detObj["z"])
                ax.scatter(Xs, Ys, Zs, c='r', marker='o', label="Radar Data")
                ax.set_xlabel('X(cm)')
                ax.set_ylabel('range (cm)')
                ax.set_zlabel('elevation (cm)')
                ax.set_xlim(-500, 500)
                ax.set_ylim(0, 500)
                ax.set_zlim(-500, 500)
                plt.draw()
                plt.pause(0.04)

            # time.sleep(0.02)

        except KeyboardInterrupt or serial.SerialException:
            radar.close_connection()
            print("\nPeace")
            break
