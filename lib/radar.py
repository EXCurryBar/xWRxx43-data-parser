import pprint
import sys
import serial
import serial.tools.list_ports
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import functools
PLOT_RANGE_IN_CM = 600


def default_kwargs(**default_kwargs_decorator):
    def actual_decorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            default_kwargs_decorator.update(kwargs)
            return fn(*args, **default_kwargs_decorator)
        return g
    return actual_decorator


class Radar:
    @default_kwargs(remove_static_noise=False, write_file=False)
    def __init__(self, config_file_name, cli_baud_rate: int, data_baud_rate: int, **kwargs):
        """
        :param cli_baud_rate (int): baud rate of the control port
        :param data_baud_rate(int): baud rate of the data port
        """
        self.args = kwargs
        # buffer-ish variable
        self._config = list()
        self._config_parameter = dict()
        self.length_list = list()
        self.xs = list()
        self.ys = list()
        self.zs = list()
        self.accumulated = dict()

        # uart things variable
        port = self._read_com_port()
        self._cli = serial.Serial(port["CliPort"], cli_baud_rate)
        self._data = serial.Serial(port["DataPort"], data_baud_rate)
        self._send_config(config_file_name)
        self._parse_config()

        # plotting variable
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # self.fig = plt.figure(figsize=(6, 6))
        # self.ax = plt.subplot(1, 1, 1)  # rows, cols, idx
        # logging things
        if self.args["write_file"]:
            self._wrote_flag = True
            self._file_name = datetime.today().strftime("%Y-%m-%d-%H%M")
            self._writer = open(f"./output_file/{self._file_name}.json", 'a', encoding="UTF-8")

        # 這個可以更大
        # self.max_buffer_size = 2**10 # 1k
        # self.max_buffer_size = 2**20 # 1M
        self.max_buffer_size = 2 ** 15
        self.byte_buffer = np.zeros(self.max_buffer_size, dtype='uint8')
        self.byte_buffer_length = 0

    def _send_config(self, config_file_name):
        self._config = open(f"./radar_config/{config_file_name}").readlines()
        for command in self._config:
            print(command)
            self._cli.write((command + '\n').encode())
            time.sleep(0.01)

    def _parse_config(self):
        chirp_end_index = 0
        chirp_start_index = 0
        adc_samples_next = 0
        loop_count = 0
        sample_rate = 0
        frequency_slope_const = 0
        adc_samples = 0
        start_frequency = 0
        idle_time = 0
        ramp_end_time = 0

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
                "DopplerBins": int(chirps_per_frame / num_tx),
                "RangeBins": int(adc_samples_next),
                "RangeResolution": (3e8 * sample_rate * 1e3) / (2 * frequency_slope_const * 1e12 * adc_samples),
                "RangeIndexToMeters": (3e8 * sample_rate * 1e3) / (
                            2 * frequency_slope_const * 1e12 * int(adc_samples_next)),
                "DopplerResolution":
                    3e8 / (2 * start_frequency * 1e9 * (idle_time + ramp_end_time) * 1e6 * int(
                        chirps_per_frame / num_tx)),
                "MaxRange": (300 * 0.9 * sample_rate) / (2 * frequency_slope_const * 1e3),
                "MaxVelocity": 3e8 / (4 * start_frequency * 1e9 * (idle_time + ramp_end_time) * 1e6 * num_tx)
            }
        )
        self.accumulated.update(
            {
                "doppler": np.zeros(
                    (self._config_parameter["DopplerBins"], self._config_parameter["RangeBins"]), dtype='float32'),
                "azimuth": np.zeros((self._config_parameter["RangeBins"], 63), dtype='float32')
            }
        )
        pprint.pprint(self._config_parameter)

    def parse_data(self):
        # header.version
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        object_struct_size = 12
        byte_vector_acc_max_size = 2 ** 15
        demo_uart_msg_detected_points = 1
        demo_uart_msg_range_profile = 2
        demo_uart_msg_azimuth_static_heat_map = 4
        demo_uart_msg_range_doppler = 5
        magic_word = [2, 1, 4, 3, 6, 5, 8, 7]

        magic_ok = 0
        data_ok = 0
        frame_number = 0
        detected_object = dict()
        azimuth_data = dict()
        range_doppler_data = dict()
        range_profile = list()
        radar_data = dict()

        # 讀資料
        read_buffer = self._data.read(self._data.in_waiting)
        byte_vector = np.frombuffer(read_buffer, dtype='uint8')
        byte_count = len(byte_vector)
        if (self.byte_buffer_length + byte_count) < self.max_buffer_size:
            self.byte_buffer[self.byte_buffer_length:self.byte_buffer_length + byte_count] = byte_vector[:byte_count]
            self.byte_buffer_length += byte_count
        # print(self.byte_buffer)
        if self.byte_buffer_length > 16:
            # possible_location = np.where(byte_vector == magic_word[0])[0]
            possible_location = np.where(self.byte_buffer == magic_word[0])[0]

            start_index = list()
            for loc in possible_location:
                # check = byte_vector[loc:loc+8]
                check = self.byte_buffer[loc:loc + 8]
                if np.array_equal(check, magic_word):
                    start_index.append(loc)

            if start_index:
                # print("start_index[0]:"+str(start_index[0]))
                if 0 < start_index[0] < self.byte_buffer_length:
                # if start_index[0] > 0:
                    try:
                        self.byte_buffer[:self.byte_buffer_length - start_index[0]] = \
                            self.byte_buffer[start_index[0]:self.byte_buffer_length]
                        self.byte_buffer_length -= start_index[0]
                        start_index[0] = 0
                    except ValueError:
                        # TODO fix here
                        pass

                if self.byte_buffer_length < 0:
                    self.byte_buffer_length = 0

                total_packet_length = np.matmul(self.byte_buffer[12:12 + 4], word)
                # print("byte_buffer_length:"+str(self.byte_buffer_length))
                # print("total_packet_length:"+str(total_packet_length))
                if (self.byte_buffer_length >= total_packet_length) and (self.byte_buffer_length != 0):
                    magic_ok = 1
                # else:
            # print(f"magic OK: {magic_ok}")
            if magic_ok:
                # return True, True, True
                index = 0
                magic_number = self.byte_buffer[index:index + 8]
                index += 8
                version = format(np.matmul(self.byte_buffer[index:index + 4], word), 'x')
                index += 4
                total_packet_length = np.matmul(self.byte_buffer[index:index + 4], word)
                index += 4
                platform = format(np.matmul(self.byte_buffer[index:index + 4], word), 'x')
                index += 4
                frame_number = np.matmul(self.byte_buffer[index:index + 4], word)
                index += 4
                time_cpu_cycle = np.matmul(self.byte_buffer[index:index + 4], word)
                index += 4
                num_detected_object = np.matmul(self.byte_buffer[index:index + 4], word)
                index += 4
                tlv_types = np.matmul(self.byte_buffer[index:index + 4], word)
                index += 8

                for _ in range(tlv_types):
                    # print(self.byte_buffer[index:index + 4])
                    tlv_type = np.matmul(self.byte_buffer[index:index + 4], word)
                    index += 4
                    tlv_length = np.matmul(self.byte_buffer[index:index + 4], word)
                    index += 4
                    if tlv_type == demo_uart_msg_detected_points:
                        tlv_num_obj = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                        # print("tlv_num_obj:", tlv_num_obj)
                        index += 2
                        tlv_xyz_format = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                        index += 2
                        range_index = np.zeros(num_detected_object, dtype='int16')
                        doppler_index = np.zeros(num_detected_object, dtype='int16')
                        peak_value = np.zeros(num_detected_object, dtype='int16')
                        x = np.zeros(num_detected_object, dtype='int16')
                        y = np.zeros(num_detected_object, dtype='int16')
                        z = np.zeros(num_detected_object, dtype='int16')

                        param_list = [range_index, doppler_index, peak_value, x, y, z]
                        for i in range(num_detected_object):
                            for item in param_list:
                                item[i] = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                                index += 2
                        range_value = range_index * self._config_parameter["RangeIndexToMeters"]
                        doppler_index[doppler_index > (self._config_parameter["DopplerBins"] / 2 - 1)] = \
                            doppler_index[doppler_index > (self._config_parameter["DopplerBins"] / 2 - 1)] - 65535
                        doppler_value = doppler_index * self._config_parameter["DopplerResolution"]
                        x = x / tlv_xyz_format
                        y = y / tlv_xyz_format
                        z = z / tlv_xyz_format
                        print(len(x), end='\r')
                        detected_object.update(
                            {
                                "NumObj": num_detected_object.tolist(),
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
                        num_bytes = 2 * self._config_parameter["RangeBins"]
                        range_profile = self.byte_buffer[index:index + num_bytes].view(dtype='int16')
                        index += tlv_length
                        data_ok = 1

                    elif tlv_type == demo_uart_msg_azimuth_static_heat_map:
                        num_tx_azim_ant = 2
                        num_rx_ant = 4
                        num_bytes = num_tx_azim_ant * num_rx_ant * self._config_parameter["RangeBins"] * 4

                        q = self.byte_buffer[index:index + num_bytes]

                        index += num_bytes
                        q_rows = num_tx_azim_ant * num_rx_ant
                        q_cols = self._config_parameter["RangeBins"]
                        num_angle_bins = 64

                        real = q[::4] + q[1::4] * 256
                        imaginary = q[2::4] + q[3::4] * 256

                        real = real.astype(np.int16)
                        imaginary = imaginary.astype(np.int16)

                        q = real + 1j * imaginary

                        q = np.reshape(q, (q_rows, q_cols), order="F")

                        Q = np.fft.fft(q, num_angle_bins, axis=0)
                        QQ = np.fft.fftshift(abs(Q), axes=0)
                        QQ = QQ.T
                        QQ = QQ[:, 1:]
                        QQ = np.fliplr(QQ).tolist()

                        theta = np.rad2deg(
                            np.arcsin(np.array(range(-num_angle_bins//2+1, num_angle_bins//2))*(2/num_angle_bins)))
                        range_array = np.arange(
                            0, self._config_parameter["RangeBins"]) * self._config_parameter["RangeIndexToMeters"]
                        # range1 -= Params.compRxChanCfg.rangeBias
                        # rangeArray = np.maximum(rangeArray, 0)
                        
                        pos_x = np.outer(range_array.T, np.sin(np.deg2rad(theta)))
                        pos_y = np.outer(range_array.T, np.cos(np.deg2rad(theta)))
                        azimuth_data.update(
                            {
                                "posX": pos_x,
                                "posY": pos_y,
                                "range": range_array,
                                "theta": theta.tolist(),
                                "heatMap": QQ
                            }
                        )
                        if self.args["remove_static_noise"]:
                            azimuth_data.update(
                                {
                                    "heatMap": self._accumulate_weight(azimuth_data["heatMap"], mode="azimuth")
                                }
                            )
                        data_ok = 1

                    elif tlv_type == demo_uart_msg_range_doppler:
                        num_bytes = 2 * self._config_parameter["RangeBins"] * self._config_parameter["DopplerBins"]

                        payload = self.byte_buffer[index:index + num_bytes]
                        index += num_bytes
                        range_doppler = payload.view(dtype=np.int16)

                        if np.max(range_doppler) > 10000:
                            continue

                        range_doppler = np.reshape(
                            range_doppler,
                            (self._config_parameter["DopplerBins"], self._config_parameter["RangeBins"]),
                            'F'
                        )

                        range_doppler = np.append(
                            range_doppler[int(len(range_doppler) / 2):],
                            range_doppler[:int(len(range_doppler) / 2)],
                            axis=0
                        )

                        range_array = np.array(
                            range(self._config_parameter["RangeBins"])) * self._config_parameter["RangeIndexToMeters"]

                        doppler_array = np.multiply(
                            np.arange(
                                -self._config_parameter["DopplerBins"] / 2,
                                self._config_parameter["DopplerBins"] / 2
                            ),
                            self._config_parameter["DopplerResolution"]
                        )
                        range_doppler_data.update(
                            {
                                "range-doppler": range_doppler.tolist(),
                                "range-array": range_array.tolist(),
                                "doppler-array": doppler_array.tolist()
                            }
                        )
                        if self.args["remove_static_noise"]:
                            range_doppler_data.update(
                                {
                                    "range-doppler": self._accumulate_weight(range_doppler_data["range-doppler"],
                                                                             mode="doppler",
                                                                             alpha=0.6,
                                                                             threshold=200)
                                }
                            )
                        data_ok = 1

                if index > 0 and data_ok == 1:
                    shift_index = index
                    try:
                        self.byte_buffer[:self.byte_buffer_length - shift_index] = \
                            self.byte_buffer[shift_index:self.byte_buffer_length]
                        self.byte_buffer_length -= shift_index
                    except ValueError:
                        pass

                    if self.byte_buffer_length < 0:
                        self.byte_buffer_length = 0

        radar_data = {
            "3d_scatter": detected_object,
            "azimuth_heatmap": azimuth_data,
            "range_doppler": range_doppler_data,
            "range_profile": range_profile
        }
        self.write_to_json(radar_data)
        return data_ok, frame_number, radar_data

    def close_connection(self):
        if self.args["write_file"]:
            self._writer.write("]")
            self._writer.close()
        self._cli.write("sensorStop\n".encode())
        time.sleep(0.5)
        self._cli.close()
        self._data.close()

    def _read_com_port(self):
        data_port = ""
        cli_port = ""
        if sys.platform == "win32" or sys.platform == "cygwin":
            ports = serial.tools.list_ports.comports(include_links=False)
            for port in ports:
                if "Enhanced COM Port" in port.description:
                    cli_port = port.name
                elif "Standard COM Port" in port.description:
                    data_port = port.name

            if not data_port or not cli_port:
                input("please connect the radar and press Enter...")
                return self._read_com_port()
            else:
                return {
                    "DataPort": data_port,
                    "CliPort": cli_port
                }
        elif sys.platform == "linux":
            return {
                "DataPort": "/dev/ttyACM1",
                "CliPort": "/dev/ttyACM0"
            }

    def plot_3d_scatter(self, detected_object):
        if self.args["remove_static_noise"]:
            self._remove_static(detected_object)
        if len(self.length_list) >= 10:  # delay x * 0.04 s
            self.xs = self.xs[self.length_list[0]:]
            self.ys = self.ys[self.length_list[0]:]
            self.zs = self.zs[self.length_list[0]:]
            self.length_list.pop(0)
        self.ax.cla()
        self.length_list.append(len(detected_object["x"]))
        self.xs += list(detected_object["x"])
        self.ys += list(detected_object["y"])
        self.zs += list(detected_object["z"])
        self.ax.scatter(self.xs, self.ys, self.zs, c='r', marker='o', label="Radar Data")
        self.ax.set_xlabel('X(cm)')
        self.ax.set_ylabel('range (cm)')
        self.ax.set_zlabel('elevation (cm)')
        self.ax.set_xlim(-PLOT_RANGE_IN_CM, PLOT_RANGE_IN_CM)
        self.ax.set_ylim(0, PLOT_RANGE_IN_CM)
        self.ax.set_zlim(-PLOT_RANGE_IN_CM, PLOT_RANGE_IN_CM)
        plt.draw()
        plt.pause(1 / 30)

    def plot_range_doppler(self, heatmap_data):
        plt.clf()
        try:
            if self.args["remove_static_noise"]:
                cs = plt.contourf(
                    heatmap_data["range-array"],
                    heatmap_data["doppler-array"],
                    heatmap_data["range-doppler"],
                    vmax=1000,
                    vmin=200
                )
            else:
                cs = plt.contourf(
                    heatmap_data["range-array"],
                    heatmap_data["doppler-array"],
                    heatmap_data["range-doppler"],
                )
            self.fig.colorbar(cs)
            self.fig.canvas.draw()
            plt.pause(0.1)
        except KeyError:
            pass

    def _accumulate_weight(self, data, mode, alpha=0.7, threshold=400):
        try:
            self.accumulated[mode] = \
                self.accumulated[mode] * (1 - alpha) + np.array(data, dtype='float32') * alpha
            plot_data = np.array(data, dtype='float32') - self.accumulated[mode]
            for i in range(len(plot_data)):
                for j in range(len(plot_data[i])):
                    if plot_data[i][j] <= threshold:
                        plot_data[i][j] = threshold

            return plot_data
        except KeyError:
            print("corrupted data")

    def plot_heat_map(self, detected_object):
        plt.clf()
        if self.args["remove_static_noise"]:
            cs = plt.contourf(
                detected_object["posX"],
                detected_object["posY"],
                detected_object["heatMap"],
                vmax=2000,
                vmin=400
            )
        else:
            cs = plt.contourf(
                detected_object["posX"],
                detected_object["posY"],
                detected_object["heatMap"],
            )
        # 绘制热力图
        self.fig.colorbar(cs)
        self.fig.canvas.draw()
        plt.pause(0.1)

    # @staticmethod
    # def plot_range_profile(range_bins):
    #     plt.clf()
    #     plt.ylim((0, 10000))
    #     plt.xlim((0, 256))
    #     plt.plot(range_bins)
    #     plt.pause(0.0001)

    @staticmethod
    def _remove_static(detected_object):
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
        detected_object.update(
            {
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
        )

    def write_to_json(self, detected_object: dict):
        for key, values in detected_object.items():
            detected_object[key] = values.tolist()
        new_line = json.dumps(detected_object)
        if self._wrote_flag:
            self._writer.write(f"[[{new_line}]")
            self._wrote_flag = False
        else:
            self._writer.write(f",\n[{new_line}]")
