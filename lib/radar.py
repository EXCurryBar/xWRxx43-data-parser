import serial
import serial.tools.list_ports
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
PLOT_RANGE_IN_CM = 500


class Radar:
    def __init__(self, config_file_name, cli_baud_rate, data_baud_rate):
        """

        :param cli_baud_rate (int): baud rate of the control port
        :param data_baud_rate(int): baud rate of the data port
        """
        # buffer-ish variable
        self._config = list()
        self._config_parameter = dict()
        self.length_list = list()
        self.xs = list()
        self.ys = list()
        self.zs = list()

        # uart things variable
        port = self._read_com_port()
        self._cli = serial.Serial(port["CliPort"], cli_baud_rate)
        self._data = serial.Serial(port["DataPort"], data_baud_rate)
        self._send_config(config_file_name)
        self._parse_config()

        # plotting variable
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # logging things
        self._wrote_flag = True
        self._file_name = datetime.today().strftime("%Y-%m-%d-%H%M")
        self._writer = open(f"./output_file/{self._file_name}.json", 'a', encoding="UTF-8")

    def _send_config(self, config_file_name):
        self._config = open(f"./radar_config/{config_file_name}").readlines()
        for command in self._config:
            print(command)
            self._cli.write((command + '\n').encode())
            time.sleep(0.01)

    def _parse_config(self):
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
                            for variable in [range_index, doppler_index, peak_value, x, y, z]:
                                variable[i] = np.matmul(byte_buffer[index:index + 2], word[:2])
                                index += 2

                        range_value = range_index * self._config_parameter["RangeIndexToMeters"]
                        doppler_index[doppler_index > (self._config_parameter["DopplerBins"]/2-1)] = \
                            doppler_index[doppler_index > (self._config_parameter["DopplerBins"]/2-1)]-65535
                        doppler_value = doppler_index * self._config_parameter["DopplerResolution"]

                        x, y, z = map(lambda item: item/tlv_xyz_format, [x, y, z])

                        detected_object.update(
                            {
                                "NumObj": tlv_num_obj.tolist(),
                                "RangeIndex": range_index.tolist(),
                                "Range": range_value.tolist(),
                                "DopplerIndex": doppler_index.tolist(),
                                "Doppler": doppler_value.tolist(),
                                "PeakValue": peak_value.tolist(),
                                "x": x.tolist(),
                                "y": y.tolist(),
                                "z": z.tolist()
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
        self._writer.write("]")
        self._writer.close()
        self._cli.write("sensorStop\n".encode())
        time.sleep(0.01)
        self._cli.close()
        self._data.close()

    def _read_com_port(self):
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
            return self._read_com_port()
        else:
            return {
                "DataPort": data_port,
                "CliPort": cli_port
            }

    def plot_3d_scatter(self, detected_object, remove_static=True):
        if len(self.length_list) >= 10:  # clear every X*0.04 s
            self.xs = self.xs[self.length_list[0]:]
            self.ys = self.ys[self.length_list[0]:]
            self.zs = self.zs[self.length_list[0]:]
            self.length_list.pop(0)
        self.ax.cla()
        if remove_static:
            detected_object = self.remove_static(detected_object)
        self.length_list.append(detected_object["NumObj"])
        self.xs += detected_object["x"]
        self.ys += detected_object["y"]
        self.zs += detected_object["z"]
        self.ax.scatter(self.xs, self.ys, self.zs, c='r', marker='o', label="Radar Data")
        self.ax.set_xlabel('azimuth (cm)')
        self.ax.set_ylabel('range (cm)')
        self.ax.set_zlabel('elevation (cm)')
        self.ax.set_xlim(-PLOT_RANGE_IN_CM, PLOT_RANGE_IN_CM)
        self.ax.set_ylim(0, PLOT_RANGE_IN_CM)
        self.ax.set_zlim(-PLOT_RANGE_IN_CM, PLOT_RANGE_IN_CM)
        plt.draw()
        plt.pause(1 / 30)

    @staticmethod
    def remove_static(detected_object):
        motion = detected_object["Doppler"]
        range_index = list(detected_object["RangeIndex"])
        range_value = list(detected_object["Range"])
        doppler_index = list(detected_object["DopplerIndex"])
        peak = list(detected_object["PeakValue"])
        xs = list(detected_object["x"])
        ys = list(detected_object["y"])
        zs = list(detected_object["z"])
        static_index = [i for i in range(len(motion)) if motion[i] == 0]
        for index in sorted(static_index, reverse=True):
            del motion[index]
            del range_index[index]
            del range_value[index]
            del doppler_index[index]
            del peak[index]
            del xs[index]
            del ys[index]
            del zs[index]
        return {
            "NumObj": len(range_index),
            "RangeIndex": range_index,
            "Range": range_value,
            "DopplerIndex": doppler_index,
            "Doppler": motion,
            "PeakValue": peak,
            "x": xs,
            "y": ys,
            "z": zs
        }

    def write_to_json(self, detected_object, remove_static=True):
        if remove_static:
            detected_object = self.remove_static(detected_object)
        new_line = json.dumps(detected_object)
        if self._wrote_flag:
            self._writer.write(f"[[{time.time()}, {new_line}]")
            self._wrote_flag = False
        else:
            self._writer.write(f",\n[{time.time()}, {new_line}]")
