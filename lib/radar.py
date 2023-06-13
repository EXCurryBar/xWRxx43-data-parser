import pprint
import sys
import serial
import serial.tools.list_ports
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

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
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.subplot(1, 1, 1)  # rows, cols, idx
        # logging things
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
        range_doppler_data = dict()
        range_profile = list()

        # 讀資料
        read_buffer = self._data.read(self._data.in_waiting)
        byte_vector = np.frombuffer(read_buffer, dtype='uint8')
        byte_count = len(byte_vector)
        # print("byte_buffer_length:"+str(self.byte_buffer_length))

        if (self.byte_buffer_length + byte_count) < self.max_buffer_size:
            # 確認讀進來的資料比buffer小
            self.byte_buffer[self.byte_buffer_length:self.byte_buffer_length + byte_count] = byte_vector[:byte_count]
            self.byte_buffer_length += byte_count
        else:
            # TODO error handle
            pass

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
                if start_index[0] > 0:
                    self.byte_buffer[:self.byte_buffer_length - start_index[0]] = \
                        self.byte_buffer[start_index[0]:self.byte_buffer_length]
                    self.byte_buffer_length -= start_index[0]
                    start_index[0] = 0

                if self.byte_buffer_length < 0:
                    self.byte_buffer_length = 0

                total_packet_length = np.matmul(self.byte_buffer[12:12 + 4], word)
                # print("byte_buffer_length:"+str(self.byte_buffer_length))
                # print("total_packet_length:"+str(total_packet_length))
                if (self.byte_buffer_length >= total_packet_length) and (self.byte_buffer_length != 0):
                    magic_ok = 1
                # else:
                #     print("magic_not_OK")
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
                tlvs = np.matmul(self.byte_buffer[index:index + 4], word)
                index += 4
                # print("frame_number:" + str(frame_number))

                for _ in range(tlvs):
                    tlv_type = np.matmul(self.byte_buffer[index:index + 4], word)
                    # print("tlvs:" + str(tlv_type))
                    index += 4
                    tlv_length = np.matmul(self.byte_buffer[index:index + 4], word)
                    index += 4
                    if tlv_type == demo_uart_msg_detected_points:
                        print("demo_uart_msg_detected_points")
                        tlv_num_obj = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                        index += 2
                        tlv_xyz_format = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                        index += 2

                        range_index = np.zeros(tlv_num_obj, dtype='int16')
                        doppler_index = np.zeros(tlv_num_obj, dtype='int16')
                        peak_value = np.zeros(tlv_num_obj, dtype='int16')
                        x = np.zeros(tlv_num_obj, dtype='int16')
                        y = np.zeros(tlv_num_obj, dtype='int16')
                        z = np.zeros(tlv_num_obj, dtype='int16')

                        for i in range(tlv_num_obj):
                            range_index[i] = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                            index += 2
                            doppler_index[i] = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                            index += 2
                            peak_value[i] = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                            index += 2
                            x[i] = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                            index += 2
                            y[i] = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                            index += 2
                            z[i] = np.matmul(self.byte_buffer[index:index + 2], word[:2])
                            index += 2

                        range_value = range_index * self._config_parameter["RangeIndexToMeters"]
                        doppler_index[doppler_index > (self._config_parameter["DopplerBins"] / 2 - 1)] = \
                            doppler_index[doppler_index > (self._config_parameter["DopplerBins"] / 2 - 1)] - 65535
                        doppler_value = doppler_index * self._config_parameter["DopplerResolution"]
                        x = x / tlv_xyz_format
                        y = y / tlv_xyz_format
                        z = z / tlv_xyz_format

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
                        num_bytes = 2 * self._config_parameter["RangeBins"]
                        range_profile = self.byte_buffer[index:index + num_bytes].view(dtype='int16')
                        index += tlv_length
                        data_ok = 1

                    elif tlv_type == demo_uart_msg_azimuth_static_heat_map:
                        numTxAzimAnt = 2
                        numRxAnt = 4
                        numBytes = numTxAzimAnt * numRxAnt * self._config_parameter["RangeBins"] * 4

                        q = self.byte_buffer[index:index + numBytes]

                        index += numBytes
                        qrows = numTxAzimAnt * numRxAnt
                        qcols = self._config_parameter["RangeBins"] 
                        NUM_ANGLE_BINS = 64

                        real = q[::4] + q[1::4] * 256
                        imaginary = q[2::4] + q[3::4] * 256

                        real = real.astype(np.int16)
                        imaginary = imaginary.astype(np.int16)

                        q = real + 1j * imaginary

                        q = np.reshape(q, (qrows, qcols), order="F")

                        Q = np.fft.fft(q,NUM_ANGLE_BINS,axis=0)
                        QQ = np.fft.fftshift(abs(Q),axes=0);
                        QQ = QQ.T
                        QQ = QQ[:,1:]
                        QQ = np.fliplr(QQ).tolist()

                        theta = np.rad2deg(
                            np.arcsin(np.array(range(-NUM_ANGLE_BINS//2+1,NUM_ANGLE_BINS//2))*(2/NUM_ANGLE_BINS)))
                        rangeArray = np.arange(
                            0, self._config_parameter["RangeBins"]) * self._config_parameter["RangeIndexToMeters"]
                        # range1 -= Params.compRxChanCfg.rangeBias
                        # rangeArray = np.maximum(rangeArray, 0)
                        
                        posX = np.outer(rangeArray.T,np.sin(np.deg2rad(theta)))
                        posY = np.outer(rangeArray.T,np.cos(np.deg2rad(theta)))
                        detected_object.update(
                            {
                                "posX": posX,
                                "posY": posY,
                                "range": rangeArray,
                                "theta": theta.tolist(),
                                "heatMap": QQ
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
                                "range-array": range_array,
                                "doppler-array": doppler_array
                            }
                        )
                        data_ok = 1

                if index > 0 and data_ok == 1:
                    shift_index = index
                    self.byte_buffer[:self.byte_buffer_length - shift_index] = \
                        self.byte_buffer[shift_index:self.byte_buffer_length]
                    self.byte_buffer_length -= shift_index

                    if self.byte_buffer_length < 0:
                        self.byte_buffer_length = 0

        return data_ok, frame_number, detected_object, range_doppler_data, range_profile

    def close_connection(self):
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
        elif sys.platform == "linux":
            return {
                "DataPort": "/dev/ttyACM1",
                "CliPort": "/dev/ttyACM0"
            }

    def plot_3d_scatter(self, detected_object):
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
        
    def plot_heat_map(self, detected_object):
        plt.clf()
        cs = plt.contourf(detected_object["posX"],detected_object["posY"],detected_object["heatMap"])
        # 绘制热力图
        self.fig.colorbar(cs)
        self.fig.canvas.draw()
        plt.pause(0.1)

    def plot_range_doppler(self, heatmap_data):
        plt.clf()
        cs = plt.contourf(
            heatmap_data["range-array"],
            heatmap_data["doppler-array"],
            heatmap_data["range-doppler"]
        )
        self.fig.colorbar(cs)
        self.fig.canvas.draw()
        plt.pause(0.1)

    @staticmethod
    def plot_range_profile(range_bins):
        plt.cla()
        plt.ylim((0, 10000))
        plt.xlim((0, 256))
        plt.plot(range_bins)
        plt.pause(0.0001)

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

    def write_to_json(self, detected_object):
        new_line = json.dumps(detected_object)
        if self._wrote_flag:
            self._writer.write(f"[[{new_line}]")
            self._wrote_flag = False
        else:
            self._writer.write(f",\n[{new_line}]")
